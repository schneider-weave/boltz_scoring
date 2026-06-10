from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from rdkit.Chem import Mol
from boltzgen.data import const
from boltzgen.data.data import Input, MSA, MSADeletion, MSAResidue, MSASequence, Target
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.pad import pad_to_max
from boltzgen.data.mol import load_canonicals, load_molecules
from boltzgen.data.parse.a3m import process_a3m
from boltzgen.data.parse.schema import YamlDesignParser
from boltzgen.data.template.features import load_dummy_templates
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.select.protein import ProteinSelector


@dataclass
class DataConfig:
    """Data configuration."""

    moldir: str
    multiplicity: int
    yaml_path: Union[List[str], str]
    tokenizer: Tokenizer
    featurizer: Featurizer
    backbone_only: bool = False
    atom14: bool = False
    atom37: bool = False
    design: bool = True
    compute_affinity: bool = False
    disulfide_prob: float = 1.0
    disulfide_on: bool = False
    skip_existing: bool = False
    skip_offset: int = 0
    diffusion_samples: int = 1
    output_dir: Optional[str] = None
    max_seqs: int = 256
  
   


@dataclass
class Dataset:
    yaml_path: Union[List[str], str]
    tokenizer: Tokenizer
    featurizer: Featurizer
    multiplicity: int = 1
    max_seqs: int = 256


def _resolve_msa_path(msa_id: Union[str, int], yaml_path: Path) -> Optional[Path]:
    """Return a concrete custom MSA path, or None for empty/auto MSA modes."""
    if msa_id in (-1, 0, None, ""):
        return None

    msa_path = Path(str(msa_id)).expanduser()
    if not msa_path.is_absolute():
        msa_path = yaml_path.parent / msa_path
    return msa_path


def _token_ids_to_sequence(token_ids: np.ndarray) -> str:
    """Convert Boltz token ids to one-letter amino-acid sequence for messages."""
    return "".join(
        const.prot_token_to_letter.get(const.tokens[int(token_id)], "X")
        for token_id in token_ids
    )


def _sequence_to_msa_residues(sequence: str) -> np.ndarray:
    """Convert a one-letter sequence to MSA residue rows."""
    token_ids = [
        const.token_ids[const.prot_letter_to_token.get(letter, "UNK")]
        for letter in sequence
    ]
    return np.array([(token_id,) for token_id in token_ids], dtype=MSAResidue)


def _gap_msa_residues(length: int) -> np.ndarray:
    """Create gap columns for MSA rows that do not cover a YAML prefix."""
    gap_id = const.token_ids["-"]
    return np.array([(gap_id,) for _ in range(length)], dtype=MSAResidue)


def _left_pad_msa(msa: MSA, prefix: str) -> MSA:
    """Pad an MSA with leading columns so its query matches a longer target."""
    prefix_len = len(prefix)
    if prefix_len == 0:
        return msa

    new_residue_chunks = []
    new_deletion_chunks = []
    new_sequences = []
    residue_offset = 0
    deletion_offset = 0

    for seq_pos, sequence in enumerate(msa.sequences):
        old_residues = msa.residues[sequence["res_start"] : sequence["res_end"]]
        prefix_residues = (
            _sequence_to_msa_residues(prefix)
            if seq_pos == 0
            else _gap_msa_residues(prefix_len)
        )
        residues = np.concatenate([prefix_residues, old_residues])
        new_residue_chunks.append(residues)

        old_deletions = msa.deletions[sequence["del_start"] : sequence["del_end"]].copy()
        if len(old_deletions) > 0:
            old_deletions["res_idx"] += prefix_len
            new_deletion_chunks.append(old_deletions)

        del_count = len(old_deletions)
        new_sequences.append(
            (
                sequence["seq_idx"],
                sequence["taxonomy"],
                residue_offset,
                residue_offset + len(residues),
                deletion_offset,
                deletion_offset + del_count,
            )
        )
        residue_offset += len(residues)
        deletion_offset += del_count

    return MSA(
        residues=np.concatenate(new_residue_chunks)
        if new_residue_chunks
        else np.array([], dtype=MSAResidue),
        deletions=np.concatenate(new_deletion_chunks)
        if new_deletion_chunks
        else np.array([], dtype=MSADeletion),
        sequences=np.array(new_sequences, dtype=MSASequence),
    )


def _validate_msa_residue_types(
    input_residues: np.ndarray,
    chain_name: str,
    msa: MSA,
    msa_path: Path,
) -> MSA:
    """Ensure the first MSA row matches the input sequence, padding prefixes if needed."""
    if len(msa.sequences) == 0:
        raise ValueError(f"Custom MSA is empty for chain {chain_name}: {msa_path}")

    first = msa.sequences[0]
    msa_residues = msa.residues[first["res_start"] : first["res_end"]]["res_type"]
    if len(input_residues) == len(msa_residues) and np.array_equal(
        input_residues,
        msa_residues,
    ):
        return msa

    input_seq = _token_ids_to_sequence(input_residues)
    msa_seq = _token_ids_to_sequence(msa_residues)
    if input_seq.endswith(msa_seq):
        prefix = input_seq[: -len(msa_seq)]
        padded = _left_pad_msa(msa, prefix)
        print(
            f"Custom MSA for chain {chain_name} is missing {len(prefix)} leading "
            f"target residues; padded those columns in memory for {msa_path}."
        )
        return padded

    raise ValueError(
        "Custom MSA first sequence must match the YAML protein sequence exactly. "
        f"Chain={chain_name}, MSA={msa_path}, "
        f"yaml_len={len(input_seq)}, msa_len={len(msa_seq)}, "
        f"yaml_start={input_seq[:30]}, msa_start={msa_seq[:30]}"
    )


def load_custom_msa_for_residue_types(
    input_residues: np.ndarray,
    chain_name: str,
    msa_path: Union[Path, str],
) -> MSA:
    """Load and validate a custom A3M for a specific structure chain."""
    msa_path = Path(msa_path).expanduser()
    if not msa_path.exists():
        raise FileNotFoundError(
            f"Custom MSA for chain {chain_name} not found: {msa_path}"
        )
    if not (msa_path.name.endswith(".a3m") or msa_path.name.endswith(".a3m.gz")):
        raise ValueError(
            f"Unsupported custom MSA format for chain {chain_name}: {msa_path}. "
            "Expected .a3m or .a3m.gz."
        )

    msa = process_a3m(msa_path)
    return _validate_msa_residue_types(input_residues, chain_name, msa, msa_path)


def _custom_msa_paths(parsed: Target, yaml_path: Path) -> Dict[int, str]:
    """Resolve custom MSA paths declared in a YAML design spec."""
    paths: Dict[int, str] = {}
    for chain in parsed.record.chains:
        msa_path = _resolve_msa_path(chain.msa_id, yaml_path)
        if msa_path is None:
            continue
        paths[chain.chain_id] = str(msa_path.resolve())
    return paths


def _load_custom_msas(parsed: Target, yaml_path: Path) -> Dict[int, MSA]:
    """Load custom A3M MSAs declared in a YAML design spec."""
    msas: Dict[int, MSA] = {}
    for chain in parsed.record.chains:
        msa_path = _resolve_msa_path(chain.msa_id, yaml_path)
        if msa_path is None:
            continue

        structure_chain = parsed.structure.chains[
            parsed.structure.chains["asym_id"] == chain.chain_id
        ]
        if len(structure_chain) != 1:
            raise ValueError(
                f"Could not resolve structure chain {chain.chain_name} for {msa_path}"
            )
        res_start = structure_chain[0]["res_idx"]
        res_end = res_start + structure_chain[0]["res_num"]
        input_residues = parsed.structure.residues[res_start:res_end]["res_type"]

        msas[chain.chain_id] = load_custom_msa_for_residue_types(
            input_residues,
            chain.chain_name,
            msa_path,
        )
        print(
            f"Loaded custom MSA for chain {chain.chain_name} "
            f"from {msa_path} ({len(msas[chain.chain_id].sequences)} sequences)."
        )
    return msas


def collate(data: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "activity_name",
            "activity_qualifier",
            "sid",
            "cid",
            "aid",
            "normalized_protein_accession",
            "pair_id",
            "record",
            "id",
            "structure",
            "tokenized",
            "structure_bonds",
            "extra_mols",
            "data_sample_idx",
            "msa_paths",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class PredictionDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        dataset: Dataset,
        canonicals: dict[str, Mol],
        moldir: str,
        backbone_only: bool = False,
        atom14: bool = False,
        atom37: bool = False,
        extra_features: Optional[List[str]] = None,
        design: bool = True,
        compute_affinity: bool = False,
        disulfide_prob: float = 1.0,
        disulfide_on: bool = False,
        skip_offset: int = 0,
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        datasets : List[Dataset]
            The datasets to sample from.

        """
        super().__init__()
        self.dataset = dataset
        self.moldir = moldir
        self.canonicals = canonicals
        self.backbone_only = backbone_only
        self.atom14 = atom14
        self.atom37 = atom37
        self.skip_offset = skip_offset
        path = dataset.yaml_path
        self.yaml_paths = [path] if isinstance(path, str) else path

        for path in self.yaml_paths:
            filename = Path(path).name
            is_hash_named_nanobody = re.search(
                r"^nb\d{4}_[0-9a-f]{8}\.yaml$",
                filename,
            )
            if re.search(r"_\d+\.yaml$", filename) and not is_hash_named_nanobody:
                raise ValueError(
                    f"Illegal YAML filename for '{str(path)}': names must not end with the pattern _\\d+\\.yaml so, e.g., the ends '_001.yaml' or '_4.yaml' are not allowed."
                    "This pattern is reserved for internal file indexing. Sorry :)"
                )
            if "_native" in filename:
                raise ValueError(
                    f"Illegal YAML filename for '{str(path)}': names must not contain '_native' because this substring is reserved for native structure companions. Sorry :)"
                )
        self.extra_features = (
            set(extra_features) if extra_features is not None else set()
        )
        self.selector = (
            ProteinSelector(  
                design_neighborhood_sizes=[2, 4, 6, 8, 10, 12, 14, 16, 18],
                substructure_neighborhood_sizes=[2, 4, 6, 8, 10, 12, 24],
                structure_condition_prob=1.0,
                distance_noise_std=1,
                run_selection=True,
                specify_binding_sites=True,
                ss_condition_prob=0.1,
                select_all=False,
                chain_reindexing=False,
            )
        )
        self.design = design
        self.compute_affinity = compute_affinity
        self.disulfide_prob = disulfide_prob
        self.disulfide_on = disulfide_on

        self.mols = {}
        self.parser = YamlDesignParser(mol_dir=self.moldir)

    def __getitem__(self, idx: int) -> Dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        path = Path(self.yaml_paths[idx % len(self.yaml_paths)])
        feat = self.get_sample(path)
        data_sample_idx = idx // len(self.yaml_paths) + self.skip_offset
        if self.dataset.multiplicity > 1:
            feat["data_sample_idx"] = data_sample_idx
        return feat

    def get_sample(self, path: Path, sample_id: Optional[str] = None) -> Dict:
        # Get itemn also needs to take a smaple id as input
        parsed = self.parser.parse_yaml(
            path, mol_dir=self.moldir, mols=self.mols
        )
        structure = parsed.structure
        design_info = parsed.design_info

        # Tokenize structure
        tokenized = self.dataset.tokenizer.tokenize(structure)

        # Transfer conditioning information that is stored in tokens
        token_to_res = tokenized.token_to_res
        tokenized.tokens["design_mask"] = design_info.res_design_mask[token_to_res]
        tokenized.tokens["binding_type"] = design_info.res_binding_type[token_to_res]
        tokenized.tokens["structure_group"] = design_info.res_structure_groups[
            token_to_res
        ]

        # Propagate design mask to obtain chain_design_mask (True whenever something is covalently bound to any residue that is in a chain that contains a design residue).
        chain_design_mask = tokenized.tokens["design_mask"].astype(bool)
        asym_id = tokenized.tokens["asym_id"]
        while True:
            design_chains = np.unique(asym_id[chain_design_mask])
            chain_propagated = np.isin(asym_id, design_chains)
            for i, j, _ in tokenized.bonds:
                if any([chain_propagated[i], chain_propagated[j]]):
                    chain_propagated[i] = True
                    chain_propagated[j] = True
            if np.equal(chain_propagated, chain_design_mask).all():
                break
            chain_design_mask = chain_propagated.astype(bool)

        # Try to find molecules in the dataset moldir if provided
        # Find missing ones in global moldir and check if all found
        molecules = {}
        molecules.update(self.canonicals)
        mol_names = set(tokenized.tokens["res_name"].tolist())
        mol_names = mol_names - set(self.canonicals.keys())
        mol_names = mol_names - set(parsed.extra_mols.keys())
        if self.moldir is not None:
            molecules.update(load_molecules(self.moldir, mol_names))

        mol_names = mol_names - set(molecules.keys())
        molecules.update(load_molecules(self.moldir, mol_names))
        molecules.update(parsed.extra_mols)

        # Finalize input data
        # NOVA: match validator — YAML msa paths are not fed into design featurization.
        # Design runs single-sequence mode (msa={}, max_seqs=1) even when YAML lists an a3m.
        input_data = Input(
            tokens=tokenized.tokens,
            bonds=tokenized.bonds,
            token_to_res=token_to_res,
            structure=structure,
            msa={},
            templates=None,
            record=parsed.record,
        )

        # Compute features
        features = self.dataset.featurizer.process(
            input_data,
            molecules=molecules,
            random=np.random.default_rng(None),
            training=False,
            max_seqs=1,
            backbone_only=self.backbone_only,
            atom14=self.atom14,
            atom37=self.atom37,
            design=self.design,
            override_method="X-RAY DIFFRACTION",
            compute_affinity=self.compute_affinity,
            disulfide_prob=self.disulfide_prob,
            disulfide_on=self.disulfide_on,
        )

        # transfer secondary structure conditioning
        ss_type = design_info.res_ss_types[token_to_res]
        features["ss_type"] = torch.from_numpy(ss_type).to(features["ss_type"])
        features["design_ss_mask"][ss_type != const.ss_type_ids["UNSPECIFIED"]] = 1

        # set chain_design_mask
        features["chain_design_mask"] = torch.from_numpy(chain_design_mask)

        # Compute template features
        templates_features = load_dummy_templates(
            tdim=1, num_tokens=len(features["res_type"])
        )
        features.update(templates_features)

        # set last necessary features
        features["idx_dataset"] = torch.tensor(1)

        # If a smaple id is provided then this should be the sample id instead of path.stem
        if sample_id is not None:
            features["id"] = sample_id
        else:
            features["id"] = path.stem
        if "structure" in self.extra_features:
            features["structure"] = structure
        if "tokenized" in self.extra_features:
            features["tokenized"] = tokenized

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        total = len(self.yaml_paths) * (self.dataset.multiplicity - self.skip_offset)
        return max(total, 0)


class FromYamlDataModule(pl.LightningDataModule):
    """DataModule for BoltzGen."""

    def __init__(
        self, cfg: DataConfig, batch_size, num_workers, pin_memory, extra_features=None
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()

        if cfg.skip_existing and cfg.output_dir is not None:
            design_dir = Path(cfg.output_dir)
            max_idx: int = -1
            if design_dir.exists():
                pattern = re.compile(r"_(\d+)(?:\.[^.]+)$")
                max_idx = max(
                    (
                        int(m.group(1))
                        for fp in design_dir.iterdir()
                        if fp.suffix in {".cif", ".pdb"}
                        and not any(s in fp.name for s in ("_native.cif", "_metadata.npz"))
                        for m in [pattern.search(fp.name)]
                        if m
                    ),
                    default=-1,
                )
            n_samples = getattr(cfg, "diffusion_samples", 1)
            cfg.skip_offset = (max_idx // max(n_samples, 1)) + 1 if max_idx >= 0 else 0
            
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate = collate

        dataset = Dataset(
            yaml_path=cfg.yaml_path,
            multiplicity=cfg.multiplicity,
            tokenizer=cfg.tokenizer,
            featurizer=cfg.featurizer,
            max_seqs=cfg.max_seqs,
        )

        # Load canonical molecules
        canonicals = load_canonicals(cfg.moldir)

        self.predict_set = PredictionDataset(
            dataset=dataset,
            canonicals=canonicals,
            moldir=Path(cfg.moldir),
            backbone_only=cfg.backbone_only,
            atom14=cfg.atom14,
            atom37=cfg.atom37,
            extra_features=extra_features,
            design=cfg.design,
            compute_affinity=cfg.compute_affinity,
            disulfide_prob=cfg.disulfide_prob,
            disulfide_on=cfg.disulfide_on,
            skip_offset=cfg.skip_offset,
        )

    def predict_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )

    def transfer_batch_to_device(
        self,
        batch: Dict,
        device: torch.device,
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> Dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.

        Returns
        -------
        np.Any
            The transferred batch.

        """
        for key in batch:
            if key not in [
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "activity_name",
                "activity_qualifier",
                "sid",
                "cid",
                "normalized_protein_accession",
                "pair_id",
                "record",
                "id",
                "structure",
                "tokenized",
                "structure_bonds",
                "extra_mols",
                "data_sample_idx",
                "msa_paths",
            ]:
                batch[key] = batch[key].to(device)
        return batch
