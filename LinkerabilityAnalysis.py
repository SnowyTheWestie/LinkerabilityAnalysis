
from __future__ import annotations

# Standard library
import random
import json
import shutil
import time
import io
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Third-party
import pandas as pd
import numpy as np
import requests
import freesasa
from tqdm import tqdm
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdmolops

# BioPython
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa



from contextlib import contextmanager


@contextmanager
def suppress_fd_stderr():
    saved_stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 140)

# -----------------------------
# Config
# -----------------------------
CACHE_DIR = Path("rcsb_cif_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RCSB_CIF = "https://files.rcsb.org/download/{pdbid}.cif"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{}"

# Common ions to exclude
COMMON_IONS = {"NA","K","CL","MG","ZN","CA","MN","CO","CU","FE","CD","NI","SR","CS","AG","BR","IOD","HG","PB","BA","AL","AU"}

# Typical cofactors / biological small molecules to exclude (expand as you see false positives)
EXCLUDE_COF = {
    # solvents/buffers/cryoprotectants
    "DMS", "GOL", "EDO", "PEG", "PE4", "PG4", "PGE", "MPD", "ACT", "ACE", "P6E", "PE8", "C8E", "P6G",
    "FMT", "TRS", "MES", "HEP", "BME", "EPE", "IPA", "EOH", "CXS",
    "BGC", "BMA", "MAN",  # some sugars often appear as additives
    "SO4", "PO4", "NO3", "CIT",  # anions sometimes appear as HET "residues"

    # ---- Added: PEG/detergent-ish crystallization junk ----
    "1PG", "1PE", "2PE", "3PE", "7PE", "PE5", "P33", "P4G", "XPE",
    "N8E", "SDS", "MPG", "MMA", "TBU",

    # ---- Added: common small additives seen as HETs ----
    "BCN", "BTB", "NHE", "PHO", "FLC",

    # nucleotides
    "ATP","ADP","AMP","GTP","GDP","GMP","CTP","CDP","CMP","UTP","UDP","UMP","A3P","ADN","MTA","A5A",

    # nicotinamide / flavins
    "NAD","NADH","NAI","NAP","NDP","FAD","FMN",

    # coenzyme A and SAM family
    "COA","ACO","SAM","SAH","S4M",

    # heme / vitamins / cofactors
    "HEM","HEME","PLP","PMP","BTN","HEC",

    # special-case you listed
    "AR6", "GSH", "NAG","IPE","NDG","IMD","DKA","LMR",

    # carbohydrates
    "GLC","BGC","GAL","BGA","MAN","BMA","FRU",
    "ARA","RIB","XYL","CAP",
    "FUC","FUL","RHA","SOR","XYP",
    "NAG","NGA","GALN","RAM","MGL",
    "SUC","MAL","TRE","LAC","CEL","SGC",
    "GLA","IDU","SIA","IDS","SGN",

    # ---- Added: sugars from your new list ----
    "MAG","BDF","AH2","P6C","G4D","OOC","9WJ","NGK","DGD","A2G","UMA",

    # amino acids
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",

    # modified amino acids
    "MSE", "SEC", "HID", "HIE", "HIP",
    "CSO", "CSD", "CME", "OCS", "MME", "SMC",
    "SEP", "TPO", "PTR", "PYL", "DDZ",

    # ---- Added: lipid/detergent-ish ligands from your new list ----
    "OLC","OLA","OLB","PLM","RET","CLR","CLA","DAO","MYS","LMT","LHG","LMG","LLP","LFA",

    # ---- Added: misc (only if you want them treated as “exclude”) ----
    "H4B","78U","9C4"
}

# Common solvent/buffer components that can show up as HET residues
EXCLUDE_SMALL = {
    "HOH","WAT","DOD","SO4","PO4","GOL","EDO","PEG","PG4","MPD","DMS","ACT","FMT","TRS","MES","HEP","BME",

    # ---- Added: small/solvent/detergent-ish from your new list ----
    "PGE","C8E","N8E","SDS","MPG","MMA","TBU",
    "1PG","1PE","2PE","3PE","7PE","PE5","P33","P4G","XPE",
    "BCN","NHE","BTB","PHO","FLC"
}

BAD_RESNAMES_DEFAULT = {
    "ZN", "MG", "CA", "NA", "K", "MN", "FE", "CU", "CO", "NI"
}

PROTEIN_LIKE_HETS = {"MSE"}

DNA_RESNAMES = {"DA", "DC", "DG", "DT"}
RNA_RESNAMES = {"A", "C", "G", "U"}

COMMON_WATERS = {"HOH", "WAT", "DOD"}


CCD_DEF_CIF = "https://files.rcsb.org/ligands/download/{ccd}.cif"

TETRA_ANGLE_DEG = 109.4712206
THETA_CH2_DEG = 54.7356103


class ElementClassifier(freesasa.Classifier):
    purePython = True

    # Simple vdW radii (Å) good enough for screening / ranking.
    RADII = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "S": 1.80,
        "P": 1.80,
        "F": 1.47,
        "CL": 1.75,
        "BR": 1.85,
        "I": 1.98,
        # common ions/metals (mostly to prevent crashes if they slip through)
        "ZN": 1.39,
        "MG": 1.73,
        "CA": 1.94,
        "NA": 2.27,
        "K": 2.75,
        "FE": 1.56,
        "CU": 1.40,
        "MN": 1.61,
        "CO": 1.52,
        "NI": 1.63,
    }

    TWO_LETTER = {"CL","BR","NA","MG","ZN","FE","CU","MN","CO","NI","CA","LI"}

    def _guess_element(self, residueName, atomName):
        rn = str(residueName).strip().upper()
        an = str(atomName).strip().upper()

        # ions sometimes show up as residueName==atomName==element
        if rn in self.RADII and an == rn:
            return rn

        # two-letter elements: CL1 -> CL, BR2 -> BR, NA -> NA ...
        if len(an) >= 2 and an[:2] in self.TWO_LETTER:
            return an[:2]

        # typical PDB/RDKit naming: CB, CA, CD1 -> element is first letter
        if len(an) >= 1 and an[0].isalpha():
            return an[0]

        return None

    def classify(self, residueName, atomName):
        el = self._guess_element(residueName, atomName)
        return "Apolar" if el == "C" else "Polar"

    def radius(self, residueName, atomName):
        el = self._guess_element(residueName, atomName)
        return float(self.RADII.get(el, -1.0)) if el else -1.0
        

# ============================
# Core utilities
# ============================

def extract_pdb_ids(csv_file, start_row, end_row):
    """
    Extract pdb_id values from a CSV file between given row numbers
    (excluding header row).

    Parameters:
        csv_file (str): Path to the CSV file
        start_row (int): First data row number (1-based, excluding header)
        end_row (int): Last data row number (1-based, excluding header)

    Returns:
        list: List of 4-character PDB ID strings
    """

    # Read CSV as strings to avoid scientific notation issues
    df = pd.read_csv(csv_file, dtype=str)

    if "pdb_id" not in df.columns:
        raise ValueError("Column 'pdb_id' not found in the CSV file.")

    # Convert to zero-based indexing
    start_idx = start_row - 1
    end_idx = end_row

    pdb_series = df.loc[start_idx:end_idx - 1, "pdb_id"]

    pdb_list = []

    for val in pdb_series:
        if pd.isna(val):
            continue

        val = val.strip()

        # Keep only valid 4-character alphanumeric PDB codes
        if re.fullmatch(r"[A-Za-z0-9]{4}", val):
            pdb_list.append(val.upper())
        else:
            print(f"Warning: Skipping invalid entry -> {val}")

    return pdb_list


def load_pdb_ids(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    tokens = re.split(r"[,\s]+", text)

    pdb_ids = []
    for token in tokens:
        token = token.strip().strip('"').strip("'").upper()
        if len(token) == 4 and token.isalnum():
            pdb_ids.append(token)

    seen = set()
    unique_ids = []
    for pdb_id in pdb_ids:
        if pdb_id not in seen:
            unique_ids.append(pdb_id)
            seen.add(pdb_id)

    return unique_ids



def ligand_coords_from_molH(molH, atom_indices):
    conf = molH.GetConformer()
    xyz = []
    for idx in atom_indices:
        pos = conf.GetAtomPosition(int(idx))
        xyz.append([pos.x, pos.y, pos.z])
    return np.asarray(xyz, dtype=float)



def write_ligand_sdf_safe(mol, out_path: str | Path) -> None:
    """
    Robust SDF writer:
      - ensures at least one conformer (adds 2D coords if none)
      - writes via SDWriter
      - raises if output is empty
    """
    out_path = Path(out_path)

    from rdkit import Chem
    from rdkit.Chem import AllChem

    if mol is None:
        raise ValueError("mol is None")

    # Copy to avoid mutating caller molecule
    m = Chem.Mol(mol)

    # Ensure there is at least one conformer (coords)
    if m.GetNumConformers() == 0:
        AllChem.Compute2DCoords(m)

    writer = Chem.SDWriter(str(out_path))
    if writer is None:
        raise RuntimeError(f"Failed to create SDWriter for {out_path}")
    writer.write(m)
    writer.close()

    # Verify non-empty file
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Wrote empty SDF: {out_path}")


def residue_to_dict(res):
    """
    Convert a Biopython Residue to a JSON-serializable dict.
    Handles common Biopython residue attributes.
    """
    try:
        # residue.get_id() -> (hetflag, resseq, icode)
        hetflag, resseq, icode = res.get_id()
        chain_id = res.get_parent().id if res.get_parent() is not None else None
        resname = getattr(res, "resname", None) or getattr(res, "get_resname", lambda: None)()
        return {
            "type": "Residue",
            "resname": str(resname) if resname is not None else None,
            "chain_id": str(chain_id) if chain_id is not None else None,
            "resseq": int(resseq) if resseq is not None else None,
            "icode": str(icode).strip() if icode is not None else None,
            "hetflag": str(hetflag) if hetflag is not None else None,
        }
    except Exception:
        # last resort: string representation
        return {"type": "Residue", "repr": repr(res)}


def make_json_safe(x):
    """
    Recursively convert objects into JSON-serializable types.
    - Biopython Residue -> dict summary
    - numpy scalars/arrays -> python scalar/list
    - sets/tuples -> lists
    - unknown objects -> repr(...)
    """
    # primitives
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    # dict
    if isinstance(x, dict):
        return {str(k): make_json_safe(v) for k, v in x.items()}

    # list/tuple/set
    if isinstance(x, (list, tuple, set)):
        return [make_json_safe(v) for v in x]

    # numpy
    try:
        import numpy as np
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
 
    if hasattr(x, "get_id") and hasattr(x, "get_parent"):
        return residue_to_dict(x)

    return repr(x)
    

def normalize_element_symbol(sym: str) -> str:
    """
    Convert CCD element symbols like 'CL', 'BR' to RDKit-friendly 'Cl', 'Br'.
    Keeps one-letter symbols uppercased ('C', 'N', ...).
    """
    s = str(sym).strip()
    if not s:
        raise ValueError("Empty element symbol")

    if len(s) == 1:
        return s.upper()
    if len(s) == 2:
        return s[0].upper() + s[1].lower()
    return s[0].upper() + s[1:].lower()

    

# ============================
# fetching / parsing / CCD helpers
# ============================

def get_json(url: str) -> dict:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_structure_cif(pdbid: str, outdir: Path = CACHE_DIR) -> Path:
    pdbid = pdbid.strip().lower()
    outdir.mkdir(parents=True, exist_ok=True)

    path = outdir / f"{pdbid}.cif"
    if path.exists():
        return path

    url = RCSB_CIF.format(pdbid=pdbid.upper())
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)

    return path


def load_structure_cif(path: Path):
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure("struct", str(path))

    
def fetch_chemcomp(ccd: str) -> dict | None:
    """
    Fetch CCD record. Returns None if not available.
    """
    try:
        return get_json(CHEMCOMP_URL.format(ccd.upper()))
    except requests.RequestException:
        return None


def chemcomp_mw(chemcomp_json: dict) -> float | None:
    cc = chemcomp_json.get("chem_comp", {})
    mw = cc.get("formula_weight")
    try:
        return float(mw) if mw is not None else None
    except Exception:
        return None


def chemcomp_is_peptide_like(chemcomp_json: dict) -> bool:
    """
    Identify peptide-like CCD entries using the 'type' field.
    """
    cc = chemcomp_json.get("chem_comp", {})
    ctype = cc.get("type", "")
    return "PEPTIDE" in ctype.upper()

    
def fetch_ccd_definition_tables(ccd_id: str):
    """
    Fetch CCD atom and bond tables for a ligand definition from the RCSB CCD.
    """
    ccd = ccd_id.upper()
    url = CCD_DEF_CIF.format(ccd=ccd)

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    mmcif_dict = MMCIF2Dict(io.StringIO(response.text))

    def as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    atom_ids = as_list(mmcif_dict.get("_chem_comp_atom.atom_id"))
    elements = as_list(mmcif_dict.get("_chem_comp_atom.type_symbol"))

    atom_id_1 = as_list(mmcif_dict.get("_chem_comp_bond.atom_id_1"))
    atom_id_2 = as_list(mmcif_dict.get("_chem_comp_bond.atom_id_2"))
    bond_order = as_list(mmcif_dict.get("_chem_comp_bond.value_order"))

    atoms = [
        {"atom_id": atom_id, "type_symbol": element}
        for atom_id, element in zip(atom_ids, elements)
    ]
    bonds = [
        {"atom_id_1": a1, "atom_id_2": a2, "value_order": order}
        for a1, a2, order in zip(atom_id_1, atom_id_2, bond_order)
    ]

    return atoms, bonds


# ============================
# Structure processing / ligand selection
# ============================

def split_protein_ligands(structure):
    """
    Return lists of protein residues, ligand residues, and water residues.
    """
    protein, ligands, waters = [], [], []

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip().upper()

                if is_aa(residue, standard=True):
                    protein.append(residue)
                elif resname in COMMON_WATERS:
                    waters.append(residue)
                elif resname not in COMMON_IONS:
                    ligands.append(residue)

    return protein, ligands, waters


def residue_heavy_atom_count(residue) -> int:
    """
    Count heavy atoms (non-H, non-D) present in the structure for one residue.
    """
    count = 0
    for atom in residue.get_atoms():
        element = (atom.element or "").strip().upper()
        if element and element not in ("H", "D"):
            count += 1
        elif not element:
            count += 1

    return count


def choose_one_ligand(
    pdbid: str,
    mw_min: float = 150.0,
    mw_max: float = 800.0,
    verbose: bool = True,
):
    """
    Select one ligand residue instance from a PDB structure.

    Selection strategy:
      1) parse mmCIF and identify ligand residues
      2) exclude water, ions, small molecules, and cofactors
      3) exclude peptide-like CCD entries
      4) keep only ligands with MW in [mw_min, mw_max]
      5) choose the remaining ligand with the most heavy atoms
    """
    cif_path = fetch_structure_cif(pdbid)
    structure = load_structure_cif(cif_path)
    protein, lig_residues, waters = split_protein_ligands(structure)

    if verbose:
        print(f"PDB {pdbid.upper()} parsed.")
        print(f"  Protein residues: {len(protein)}")
        print(f"  Ligand residues (pre-filter): {len(lig_residues)}")
        print(f"  Waters: {len(waters)}")

    rows = []
    for res in lig_residues:
        resname = res.get_resname().strip().upper()

        if resname in EXCLUDE_SMALL or resname in EXCLUDE_COF or resname in COMMON_IONS:
            continue

        ccj = fetch_chemcomp(resname)
        if ccj is None:
            continue

        if chemcomp_is_peptide_like(ccj):
            continue

        mw = chemcomp_mw(ccj)
        if mw is None or not (mw_min <= mw <= mw_max):
            continue

        chain_id = res.get_parent().id
        hetflag, resseq, icode = res.id

        rows.append({
            "pdb_id": pdbid.upper(),
            "resname": resname,
            "chain": chain_id,
            "resseq": resseq,
            "icode": (icode.strip() if isinstance(icode, str) else str(icode)),
            "mw": mw,
            "heavy_atoms_in_structure": residue_heavy_atom_count(res),
            "bio_res_id": res.id,
            "res_obj": res,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        if verbose:
            print("No ligand candidates remain after filters (MW + exclusions + peptide check).")
        return None, df, structure

    df = df.sort_values(
        ["heavy_atoms_in_structure", "mw"],
        ascending=False,
    ).reset_index(drop=True)

    if verbose:
        print(df.drop(columns=["res_obj"]).to_string(index=False))

    chosen = df.iloc[0].to_dict()
    return chosen, df, structure



# ============================
# RDK ligand construction
# ============================

    
def rdkit_from_ccd_with_crystal_coords(ccd_id: str, lig_res):
    """
    Build an RDKit molecule from CCD atom/bond definitions and assign
    crystal coordinates from the ligand residue in the structure.

    Parameters
    ----------
    ccd_id : str
        CCD ligand identifier.
    lig_res
        Biopython residue object for the ligand instance.

    Returns
    -------
    tuple
        molH : RDKit molecule with explicit hydrogens
        missing : list of CCD atom IDs not found in the crystal residue
    """
    ccd_id = ccd_id.upper()

    atoms, bonds = fetch_ccd_definition_tables(ccd_id)
    if not atoms:
        raise ValueError(f"CCD {ccd_id}: no atom table found")
    if not bonds:
        raise ValueError(f"CCD {ccd_id}: no bond table found")

    rw = Chem.RWMol()
    atom_id_to_idx = {}

    for atom_row in atoms:
        atom_id = str(atom_row["atom_id"])
        element = normalize_element_symbol(atom_row["type_symbol"])
        idx = rw.AddAtom(Chem.Atom(element))
        atom_id_to_idx[atom_id] = idx
        rw.GetAtomWithIdx(idx).SetProp("ccd_atom_id", atom_id)

    def bondtype_from_value_order(value_order):
        value_order = (value_order or "").upper()
        if value_order in ("SING", "SINGLE"):
            return Chem.BondType.SINGLE, False
        if value_order in ("DOUB", "DOUBLE"):
            return Chem.BondType.DOUBLE, False
        if value_order in ("TRIP", "TRIPLE"):
            return Chem.BondType.TRIPLE, False
        if value_order in ("AROM", "AROMATIC"):
            return Chem.BondType.AROMATIC, True
        return Chem.BondType.SINGLE, False

    for bond_row in bonds:
        atom_id_1 = str(bond_row["atom_id_1"])
        atom_id_2 = str(bond_row["atom_id_2"])
        bond_type, is_aromatic = bondtype_from_value_order(bond_row["value_order"])

        idx1 = atom_id_to_idx[atom_id_1]
        idx2 = atom_id_to_idx[atom_id_2]
        rw.AddBond(idx1, idx2, bond_type)

        if is_aromatic:
            rw.GetAtomWithIdx(idx1).SetIsAromatic(True)
            rw.GetAtomWithIdx(idx2).SetIsAromatic(True)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)

    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)

    crystal_coords = {
        atom.get_name().strip(): tuple(map(float, atom.coord))
        for atom in lig_res.get_atoms()
    }

    missing = []
    for atom in mol.GetAtoms():
        atom_id = atom.GetProp("ccd_atom_id")
        if atom_id in crystal_coords:
            x, y, z = crystal_coords[atom_id]
            conf.SetAtomPosition(atom.GetIdx(), Point3D(x, y, z))
        else:
            missing.append(atom_id)

    mol.AddConformer(conf, assignId=True)

    molH = Chem.AddHs(mol, addCoords=True)
    return molH, missing


# ============================
# Modifiable atoms & site construction
# ============================


def modifiable_atoms_and_sites(molH: Chem.Mol) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify modifiable heavy atoms and individual hydrogen substitution sites.
    """
    rows_atoms = []
    rows_sites = []

    for atom in molH.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue

        nH = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1)
        if nH == 0:
            continue

        atom_name = atom.GetProp("ccd_atom_id") if atom.HasProp("ccd_atom_id") else f"atom_{atom.GetIdx()}"

        rows_atoms.append({
            "attach_atom_index": atom.GetIdx(),
            "attach_atom_name": atom_name,
            "attach_element": atom.GetSymbol(),
            "num_attached_H": nH,
            "is_aromatic": bool(atom.GetIsAromatic()),
            "in_ring": bool(atom.IsInRing()),
            "degree": int(atom.GetDegree()),
            "is_methyl_carbon": bool(atom.GetAtomicNum() == 6 and nH == 3),
        })

        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                rows_sites.append({
                    "h_atom_index": nbr.GetIdx(),
                    "attach_atom_index": atom.GetIdx(),
                    "attach_atom_name": atom_name,
                    "attach_element": atom.GetSymbol(),
                })

    df_atoms = pd.DataFrame(rows_atoms)
    if not df_atoms.empty:
        df_atoms = df_atoms.sort_values(
            ["num_attached_H", "attach_element", "attach_atom_name"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    df_sites = pd.DataFrame(rows_sites)
    if not df_sites.empty:
        df_sites = df_sites.sort_values(
            ["attach_atom_name", "h_atom_index"],
            ascending=[True, True],
        ).reset_index(drop=True)

    return df_atoms, df_sites


# ============================
# Structure cleaning + SASA pipeline
# ============================


def structure_without_bad_residues(structure, bad_resnames=BAD_RESNAMES_DEFAULT):
    """
    Return a copy of the structure with selected problematic residues removed.

    This is mainly used to remove metal ions that can interfere with SASA
    calculations while preserving the protein-ligand context.
    """
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model

    new_structure = Structure(structure.id)

    for model in structure:
        new_model = Model(model.id)
        new_structure.add(new_model)

        for chain in model:
            new_chain = chain.copy()
            new_chain.detach_parent()
            new_model.add(new_chain)

            for residue in list(new_chain):
                resname = residue.get_resname().strip().upper()
                if resname in bad_resnames:
                    new_chain.detach_child(residue.id)

    return new_structure



def freesasa_df_all_atoms(structure, classifier=None, hetatm=True, skip_unknown=True) -> pd.DataFrame:
    """
    Compute per-atom SASA for all atoms in a Bio.PDB structure.
    Returns df_all with chain/resname/resseq/atom_name/sasa_A2.
    """
    if classifier is None:
        classifier = ElementClassifier()

    options = {"hetatm": bool(hetatm), "skip-unknown": bool(skip_unknown)}

    fs_struct = freesasa.structureFromBioPDB(structure, classifier=classifier, options=options)
    with suppress_fd_stderr():
        result = freesasa.calc(fs_struct)

    rows = []
    for i in range(fs_struct.nAtoms()):
        rows.append({
            "i": i,
            "chain": fs_struct.chainLabel(i).strip(),
            "resname": fs_struct.residueName(i).strip(),
            "resseq": str(fs_struct.residueNumber(i)).strip(),
            "atom_name": fs_struct.atomName(i).strip(),
            "sasa_A2": float(result.atomArea(i)),
        })

    return pd.DataFrame(rows)

    

def ligand_sasa_from_df_all(df_all: pd.DataFrame, chosen: dict) -> pd.DataFrame:
    lig_chain  = str(chosen["chain"]).strip()
    lig_resname = str(chosen["resname"]).strip().upper()
    lig_resseq = str(chosen["resseq"]).strip()
    """
    Extract ligand atoms from full SASA table based on selected ligand instance.
    """

    lig = df_all[
        (df_all["chain"] == lig_chain) &
        (df_all["resname"].str.upper() == lig_resname) &
        (df_all["resseq"].astype(str).str.strip() == lig_resseq)
    ].copy().reset_index(drop=True)

    lig["atom_name_norm"] = lig["atom_name"].astype(str).str.strip()
    return lig




def add_sasa_to_modifiable_atoms(df_atoms: pd.DataFrame, lig_sasa: pd.DataFrame) -> pd.DataFrame:
    out = df_atoms.copy()
    out["attach_atom_name_norm"] = out["attach_atom_name"].astype(str).str.strip()

    merged = out.merge(
        lig_sasa[["atom_name_norm", "sasa_A2"]],
        left_on="attach_atom_name_norm",
        right_on="atom_name_norm",
        how="left"
    ).drop(columns=["atom_name_norm"])

    total = float(merged["sasa_A2"].sum(skipna=True)) if "sasa_A2" in merged else 0.0
    merged["ligand_total_sasa_A2"] = total
    merged["ligand_atom_sasa_fraction"] = merged["sasa_A2"] / total if total > 0 else pd.NA
    return merged


def add_sasa_to_sites(df_sites: pd.DataFrame, df_atoms_with_sasa: pd.DataFrame) -> pd.DataFrame:
    return df_sites.merge(
        df_atoms_with_sasa[["attach_atom_index", "sasa_A2", "ligand_atom_sasa_fraction", "ligand_total_sasa_A2"]],
        on="attach_atom_index",
        how="left"
    )
    

def summarize_exposure(df_atoms_with_sasa: pd.DataFrame, exposed_A2: float = 5.0) -> dict:
    """
    Simple starter summary. You can tune exposed_A2 later.
    """
    if df_atoms_with_sasa is None or df_atoms_with_sasa.empty:
        return {"exposed_A2": exposed_A2, "n_mod_atoms": 0}

    s = df_atoms_with_sasa.copy()
    s["is_exposed"] = s["sasa_A2"] > float(exposed_A2)

    return {
        "exposed_A2": float(exposed_A2),
        "n_mod_atoms": int(len(s)),
        "n_exposed_mod_atoms": int(s["is_exposed"].sum()),
        "frac_exposed_mod_atoms": float(s["is_exposed"].mean()),
        "ligand_total_sasa_A2": float(s["ligand_total_sasa_A2"].iloc[0]) if "ligand_total_sasa_A2" in s else None,
        "nan_sasa_fraction": float(s["sasa_A2"].isna().mean()),
    }


# ============================
# Single PDB-pipeline
# ============================

 
def analyze_pdb_ligand_exposure(
    pdbid: str,
    mw_min: float = 150.0,
    mw_max: float = 800.0,
    exposed_A2: float = 5.0,
    bad_resnames=BAD_RESNAMES_DEFAULT,
    verbose: bool = False,
):
    """
    Analyze ligand exposure for a single PDB entry.

    Returns
    -------
    dict
        On success:
        {
            "status": "ok",
            "pdb_id": ...,
            "chosen": ...,
            "molH": ...,
            "structure_clean": ...,
            "atoms": ...,
            "sites": ...,
            "lig_sasa": ...,
            "summary": ...,
        }

        On failure:
        {
            "status": "failed",
            "pdb_id": ...,
            "reason": "...",
            "chosen": None,
            "atoms": None,
            "sites": None,
            "lig_sasa": None,
            "summary": None,
        }
    """
    pdbid = str(pdbid).upper().strip()

    try:
        chosen, df_candidates, structure = choose_one_ligand(
            pdbid, mw_min=mw_min, mw_max=mw_max, verbose=verbose
        )

        if chosen is None:
            return {
                "status": "failed",
                "pdb_id": pdbid,
                "reason": "no_ligand_candidates_after_filters",
                "chosen": None,
                "atoms": None,
                "sites": None,
                "lig_sasa": None,
                "summary": None,
            }

        lig_res = chosen["res_obj"]
        molH, missing = rdkit_from_ccd_with_crystal_coords(chosen["resname"], lig_res)

        if molH is None or molH.GetNumAtoms() == 0:
            raise ValueError("Failed to build RDKit ligand (molH).")

        df_atoms, df_sites = modifiable_atoms_and_sites(molH)

        structure_clean = structure_without_bad_residues(
            structure,
            bad_resnames=bad_resnames,
        )

        df_all = freesasa_df_all_atoms(
            structure_clean,
            classifier=ElementClassifier(),
            hetatm=True,
            skip_unknown=True,
        )
        lig_sasa = ligand_sasa_from_df_all(df_all, chosen)

        if lig_sasa.empty:
            return {
                "status": "failed",
                "pdb_id": pdbid,
                "reason": "ligand_not_found_in_freesasa_table_after_filter",
                "chosen": chosen,
                "atoms": None,
                "sites": None,
                "lig_sasa": lig_sasa,
                "summary": None,
            }

        df_atoms_with_sasa = add_sasa_to_modifiable_atoms(df_atoms, lig_sasa)
        df_sites_with_sasa = add_sasa_to_sites(df_sites, df_atoms_with_sasa)

        summary = summarize_exposure(df_atoms_with_sasa, exposed_A2=exposed_A2)

        if verbose:
            summary["missing_ccd_atoms_in_crystal_mapping"] = missing

        return {
            "status": "ok",
            "pdb_id": pdbid,
            "chosen": chosen,
            "molH": molH,
            "structure_clean": structure_clean,
            "atoms": df_atoms_with_sasa,
            "sites": df_sites_with_sasa,
            "lig_sasa": lig_sasa,
            "summary": summary,
        }

    except Exception as e:
        return {
            "status": "failed",
            "pdb_id": pdbid,
            "reason": f"{type(e).__name__}: {e}",
            "chosen": None,
            "atoms": None,
            "sites": None,
            "lig_sasa": None,
            "summary": None,
        }



# ============================
# Substituent direction geometry
# ============================

def _unit_local(v, eps=1e-9):
    norm = float(np.linalg.norm(v))
    return None if norm < eps else (v / norm)


def _orthonormal_basis_local(v_unit):
    """
    Return (e1, e2), an orthonormal basis spanning the plane perpendicular
    to the input unit vector.
    """
    v = _unit_local(v_unit)
    if v is None:
        return None, None

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, v)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = _unit_local(np.cross(v, ref))
    e2 = _unit_local(np.cross(v, e1))
    return e1, e2




def trigonal_planar_directions_for_atom(molH, conf, attach_idx: int):
    """
    Return candidate substituent direction(s) for an sp2 / aromatic atom
    with at least one attached hydrogen.

    Uses an in-plane approximation:
      u = -unit(sum of heavy-neighbor bond unit vectors)
    """
    atomA = molH.GetAtomWithIdx(int(attach_idx))

    nH = sum(1 for nb in atomA.GetNeighbors() if nb.GetAtomicNum() == 1)
    if nH == 0:
        return []

    Apos = conf.GetAtomPosition(int(attach_idx))
    A = np.array([Apos.x, Apos.y, Apos.z], dtype=float)

    heavy_nbrs = [nb for nb in atomA.GetNeighbors() if nb.GetAtomicNum() != 1]
    if len(heavy_nbrs) == 0:
        return []

    vecs = []
    for nb in heavy_nbrs:
        p = conf.GetAtomPosition(nb.GetIdx())
        vec = np.array([p.x, p.y, p.z], dtype=float) - A
        u = _unit_local(vec)
        if u is not None:
            vecs.append(u)

    if not vecs:
        return []

    u_missing = _unit_local(-np.sum(vecs, axis=0))
    return [u_missing] if u_missing is not None else []


def linear_directions_for_atom(molH, conf, attach_idx: int):
    """
    Return candidate substituent direction(s) for an sp atom with at least
    one attached hydrogen.

    For a terminal sp center with one heavy neighbor, return the direction
    opposite that bond. Otherwise return [].
    """
    atomA = molH.GetAtomWithIdx(int(attach_idx))

    nH = sum(1 for nb in atomA.GetNeighbors() if nb.GetAtomicNum() == 1)
    if nH == 0:
        return []

    Apos = conf.GetAtomPosition(int(attach_idx))
    A = np.array([Apos.x, Apos.y, Apos.z], dtype=float)

    heavy_nbrs = [nb for nb in atomA.GetNeighbors() if nb.GetAtomicNum() != 1]
    if len(heavy_nbrs) == 1:
        p = conf.GetAtomPosition(heavy_nbrs[0].GetIdx())
        bond_vec = _unit_local(np.array([p.x, p.y, p.z], dtype=float) - A)
        if bond_vec is None:
            return []
        u = _unit_local(-bond_vec)
        return [u] if u is not None else []

    return []



def tetrahedral_directions_for_atom(molH, conf, attach_idx: int):
    """
    Return candidate substituent unit vectors for a modifiable heavy atom
    using idealized tetrahedral geometry.

    This does not use RDKit hydrogen coordinates directly.
    """
    atomA = molH.GetAtomWithIdx(int(attach_idx))

    nH = sum(1 for nb in atomA.GetNeighbors() if nb.GetAtomicNum() == 1)
    if nH == 0:
        return []

    Apos = conf.GetAtomPosition(int(attach_idx))
    A = np.array([Apos.x, Apos.y, Apos.z], dtype=float)

    heavy_nbrs = [nb for nb in atomA.GetNeighbors() if nb.GetAtomicNum() != 1]
    vecs = []
    for nb in heavy_nbrs:
        p = conf.GetAtomPosition(nb.GetIdx())
        vec = np.array([p.x, p.y, p.z], dtype=float) - A
        u = _unit_local(vec)
        if u is not None:
            vecs.append(u)

    # CH (3 heavy neighbors)
    if nH == 1 and len(vecs) == 3:
        u = _unit_local(-(vecs[0] + vecs[1] + vecs[2]))
        return [u] if u is not None else []

    # CH2 (2 heavy neighbors)
    if nH == 2 and len(vecs) == 2:
        v1, v2 = vecs
        bisector = _unit_local(v1 + v2)
        normal = _unit_local(np.cross(v1, v2))
        if bisector is None or normal is None:
            return []

        theta = np.deg2rad(THETA_CH2_DEG)
        u1 = _unit_local(-bisector * np.cos(theta) + normal * np.sin(theta))
        u2 = _unit_local(-bisector * np.cos(theta) - normal * np.sin(theta))
        return [u for u in (u1, u2) if u is not None]

    # CH3 (1 heavy neighbor)
    if nH == 3 and len(vecs) == 1:
        bond = vecs[0]
        e1, e2 = _orthonormal_basis_local(bond)
        if e1 is None or e2 is None:
            return []

        phi = np.deg2rad(TETRA_ANGLE_DEG)
        dirs = []
        for angle_deg in (0.0, 120.0, 240.0):
            angle = np.deg2rad(angle_deg)
            ring = e1 * np.cos(angle) + e2 * np.sin(angle)
            u = _unit_local(-bond * np.cos(phi) + ring * np.sin(phi))
            if u is not None:
                dirs.append(u)
        return dirs

    # Fallback
    if len(vecs) >= 1:
        u = _unit_local(-np.sum(vecs, axis=0))
        return [u] if u is not None else []

    return []

    
def substituent_directions_for_atom(molH, conf, attach_idx: int):
    """
    Return candidate unit vectors for placing a substituent on an atom that has >=1 H.
    Chooses direction model based on hybridization/aromaticity.

    - sp3: tetrahedral_directions_for_atom (existing)
    - sp2/aromatic: trigonal_planar_directions_for_atom (new)
    - sp: linear_directions_for_atom (new; less important)
    """
    atom = molH.GetAtomWithIdx(int(attach_idx))
    hyb = atom.GetHybridization()
    is_arom = atom.GetIsAromatic()

    if is_arom or hyb == Chem.rdchem.HybridizationType.SP2:
        return trigonal_planar_directions_for_atom(molH, conf, attach_idx)

    if hyb == Chem.rdchem.HybridizationType.SP:
        return linear_directions_for_atom(molH, conf, attach_idx)

    return tetrahedral_directions_for_atom(molH, conf, attach_idx)



# ============================
# protein obstacle extraction
# ============================


def get_obstacle_heavy_atom_coords(
    structure,
    chosen=None,
    exclude_waters: bool = True,
    exclude_ligand_instance: bool = True,
    exclude_hydrogens: bool = True,
    keep_only_polymer: bool = True,
    altloc_policy: str = "A_or_blank",  
):
    """
    Build a coordinate array of obstacle atoms for clash checking (typically protein heavy atoms).

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        Bio.PDB structure (e.g., structure_clean).
    chosen : dict or None
        Your chosen ligand locator dict with keys: chain, resname, resseq, icode.
        Used only if exclude_ligand_instance=True.
    exclude_waters : bool
        Drop HOH/WAT waters.
    exclude_ligand_instance : bool
        Exclude the selected ligand residue instance from obstacles.
    exclude_hydrogens : bool
        Drop atoms whose element is H / D.
    keep_only_polymer : bool
        If True, only include standard polymer residues (protein). This is conservative.
        If False, includes all non-water residues except chosen ligand if excluded.
    altloc_policy : str
        "A_or_blank": keep atoms with altloc 'A' or ' ' only.
        "highest_occupancy": for disordered atoms, keep the child with highest occupancy.

    Returns
    -------
    coords : np.ndarray, shape (N, 3)
    meta : pd.DataFrame with columns [chain, resname, resseq, icode, atom_name, element, occupancy]
    """
    
    water_resnames = COMMON_WATERS

    # Helper: identify the chosen ligand residue instance
    chosen_id = None
    if chosen and exclude_ligand_instance:
        chosen_id = (str(chosen["chain"]), str(chosen["resname"]), int(chosen["resseq"]), str(chosen.get("icode", "") or ""))

    coords_list = []
    meta_rows = []

    # Helper: polymer residue heuristic (Bio.PDB uses hetflag in residue.id[0])
    def is_polymer_residue(residue):
        hetflag = residue.id[0]  # ' ' for standard polymer, 'H_' for hetero, 'W' for water (often)
        return hetflag == " "

    # For highest occupancy selection on disordered atoms
    def pick_atom_for_altloc(atom):
        # atom can be DisorderedAtom
        # pick child atom with highest occupancy (fallback to first if all None)
        best = None
        best_occ = -1.0
        for child in atom.child_dict.values():
            occ = child.get_occupancy()
            if occ is None:
                occ = -1.0
            if occ > best_occ:
                best_occ = occ
                best = child
        return best if best is not None else atom

    for model in structure:
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                resname = residue.get_resname().strip()

                # exclude waters
                if exclude_waters and resname in water_resnames:
                    continue

                # keep only polymer residues (protein), if requested
                if keep_only_polymer and not is_polymer_residue(residue):
                    continue

                # exclude chosen ligand instance (if present in polymer=False mode typically)
                if chosen_id is not None:
                    resseq = int(residue.id[1])
                    icode = (residue.id[2] or "").strip()
                    if (str(chain_id), str(resname), resseq, str(icode)) == chosen_id:
                        continue

                for atom in residue.get_unpacked_list():
                    # altloc handling
                    if altloc_policy == "A_or_blank":
                        altloc = atom.get_altloc()
                        if altloc not in (" ", "A"):
                            continue
                        atom_use = atom
                    elif altloc_policy == "highest_occupancy":
                        atom_use = pick_atom_for_altloc(atom) if atom.is_disordered() else atom
                    else:
                        raise ValueError(f"Unknown altloc_policy: {altloc_policy}")

                    element = (atom_use.element or "").strip().upper()
                    if exclude_hydrogens and element in ("H", "D"):
                        continue

                    # coordinates
                    xyz = atom_use.get_coord()
                    if xyz is None or len(xyz) != 3:
                        continue

                    coords_list.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

                    meta_rows.append({
                        "chain": str(chain_id),
                        "resname": str(resname),
                        "resseq": int(residue.id[1]),
                        "icode": (residue.id[2] or "").strip(),
                        "atom_name": atom_use.get_name().strip(),
                        "element": element,
                        "occupancy": atom_use.get_occupancy(),
                    })

    coords = np.array(coords_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return coords, meta

    
# ============================
# cone fitting
# ============================



try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAVE_KDTREE = True
except Exception:
    cKDTree = None
    _HAVE_KDTREE = False


def _normalize_cone(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize vector(s) along the last axis.
    """
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _fibonacci_sphere_cone(n: int) -> np.ndarray:
    """
    Deterministic approximately uniform unit vectors on the sphere.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    golden = (1.0 + 5.0**0.5) / 2.0
    theta = 2.0 * np.pi * i / golden

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    return np.stack([x, y, z], axis=1).astype(np.float64)


def _orthonormal_basis_cone(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return two orthonormal vectors spanning the plane perpendicular to u.
    """
    u = np.asarray(u, dtype=np.float64).reshape(3,)
    u = _normalize_cone(u.reshape(1, 3))[0]

    if abs(u[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    v = np.cross(u, ref)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        raise ValueError("Could not construct orthonormal basis for cone axis.")
    v = v / v_norm

    w = np.cross(u, v)
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-12:
        raise ValueError("Could not construct second orthonormal basis vector for cone axis.")
    w = w / w_norm

    return v, w


   
def determine_dimension_cone(
    obstacle_coords: np.ndarray,                # Array of atoms in protein and ligand
    first_linker_atom_xyz: np.ndarray,          # Coordinates of atom at origin of cone
    *,
    r_frustum_base_A: float = 2.2,              # GEOMETRY: sets h0 and tip so bottom radius fits the root/group
    r_clash_A: float = 2.0,                     # CLEARANCE: used for blockage checks (effective obstacle inflation)
    r_solvent_shell_A: float = 2.0,            # Extra radius around upper frustum for solvent accessibility
    use_kdtree_if_available: bool = True,
    Dmax_A: float = 10.0,                       # Maximum height to be scanned in phase 2
    n_dirs: int = 96,                           # Number of points created on the Fibonacci sphere
    n_axes: int = 96,                           # Number of evaluated axes
    step_A: float = 0.5,                        # Axial increment used in height determination
    tolerate_blocked_frac: float = 0.10,        # kept for compatibility; unused in solvent-only Phase B
    alpha_grid_deg: np.ndarray | None = None,   # tested cone angles
    interior_frac: float = 0.05,                # kept for compatibility; unused in solvent-only Phase B
    target_open_frac: float = 0.90,             # kept for compatibility; used as fallback if needed
    rng_seed: int = 0,
):
    """
    Drop-in version incorporating:
      #1  Cap-sampling around an estimated "escape direction" (PCA of nearby obstacles).
      #3  Continuous ray/segment obstruction test (min distance to ray segment) instead of ray marching.

    Return tuple layout (unchanged):
      (height_from_bottom, height_clear, best_alpha_deg, tip_xyz, best_u,
       sphere_center_xyz, sphere_radius_R, height_capped)

    Notes:
      - Phase A chooses (u, alpha) by openness tests on a finite ray length.
      - Phase B (THIS VERSION) is SOLVENT-ONLY: it does NOT shorten height due to cone clashes;
        it only finds the first height where the outer "solvent shell" becomes sufficiently open.
      - This function assumes obstacle_coords are point obstacles inflated by r_clash_A (distance < r_clash_A blocks).
    """
    # ---------------------------
    # Phase A openness requirement schedule (alpha-dependent)
    # ---------------------------
    OPEN_MIN = 0.65
    OPEN_MAX = 0.90
    OPEN_GAMMA = 1.5

    # ---------------------------
    # Phase A parameters
    # ---------------------------
    RAY_LENGTH_A_FULL = 7.5
    RAY_STEP_A_FULL = 1.5     
    N_SURFACE_FULL = 72
    N_INTERIOR_FULL = 16

    RAY_LENGTH_A_COARSE = 7.5
    RAY_STEP_A_COARSE = 2.5   
    N_SURFACE_COARSE = 24
    N_INTERIOR_COARSE = 0     

    COARSE_MARGIN = 0.06

    # ---------------------------
    # Cap sampling parameters (#1)
    # ---------------------------
    CAP_ANGLE_DEG = 60.0
    ESCAPE_K_NEIGHBORS = 96
    ESCAPE_RADIUS_A = 10.0

    # ---------------------------
    # Phase B parameters
    # ---------------------------
    Dmax_A = float(Dmax_A)
    step_A = float(step_A)
    n_steps = max(1, int(np.ceil(Dmax_A / step_A)))

    bottom = np.asarray(first_linker_atom_xyz, dtype=np.float64).reshape(3,)
    obs_all = np.asarray(obstacle_coords, dtype=np.float64).reshape(-1, 3)

    # Default alpha grid
    if alpha_grid_deg is None:
        alpha_grid_deg = np.concatenate(
            [
                np.linspace(6.0, 15.0, 6),
                np.linspace(18.0, 45.0, 10),
                np.linspace(50.0, 80.0, 5),
            ]
        ).astype(np.float64)
    else:
        alpha_grid_deg = np.asarray(alpha_grid_deg, dtype=np.float64).ravel()

    alpha_grid_deg = np.sort(alpha_grid_deg)[::-1]
    alpha_hi = float(alpha_grid_deg[0])
    alpha_lo = float(alpha_grid_deg[-1])

    def _open_req_for_alpha(alpha_deg: float) -> float:
        if alpha_hi <= alpha_lo + 1e-9:
            return float(np.clip(float(target_open_frac), 0.0, 1.0))
        x = (float(alpha_deg) - alpha_lo) / (alpha_hi - alpha_lo)
        x = float(np.clip(x, 0.0, 1.0))
        return float(OPEN_MIN + (OPEN_MAX - OPEN_MIN) * (x ** float(OPEN_GAMMA)))

    # ---------------------------
    # Truncation: start outside origin atom
    # ---------------------------
    delta_trunc_A = max(0.25, 0.25 * float(r_clash_A))
    t_trunc = float(r_frustum_base_A) + float(delta_trunc_A)

    # ---------------------------
    # Ignore immediate surrounding atoms near origin
    # ---------------------------
    obs = obs_all
    if obs.shape[0] > 0:
        d0 = np.linalg.norm(obs - bottom[None, :], axis=1)
        ignore_R = float(r_frustum_base_A) + float(r_clash_A)
        obs = obs[d0 >= ignore_R]

    if obs.shape[0] == 0:
        best_alpha_deg = float(alpha_grid_deg[0])
        alpha = float(np.deg2rad(best_alpha_deg))
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        tan_a = float(np.tan(alpha))
        h0 = float(r_frustum_base_A / tan_a) if tan_a > 1e-9 else float("inf")
        tip = bottom - u * h0
        D = float(Dmax_A)
        return (float(h0), float(D), float(best_alpha_deg), tip, u, tip.copy(), float(h0 + D), True)

    # ---------------------------
    # KDTree
    # ---------------------------
    tree = None
    if use_kdtree_if_available and _HAVE_KDTREE and obs.shape[0] > 0:
        tree = cKDTree(obs)

    rng = np.random.default_rng(int(rng_seed))

    # ---------------------------
    # Estimate escape direction (PCA of nearby obstacles)
    # ---------------------------
    def _estimate_escape_direction() -> np.ndarray:
        if obs.shape[0] == 0:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        if tree is not None:
            idx = tree.query_ball_point(bottom, r=float(ESCAPE_RADIUS_A))
            if idx and len(idx) >= 8:
                nbr = obs[np.asarray(idx, dtype=np.int64)]
            else:
                k = min(int(ESCAPE_K_NEIGHBORS), obs.shape[0])
                _, ii = tree.query(bottom, k=k)
                ii = np.asarray(ii, dtype=np.int64).ravel()
                nbr = obs[ii]
        else:
            d2 = np.sum((obs - bottom[None, :]) ** 2, axis=1)
            mask = d2 <= float(ESCAPE_RADIUS_A) ** 2
            if np.any(mask):
                nbr = obs[mask]
            else:
                k = min(int(ESCAPE_K_NEIGHBORS), obs.shape[0])
                ii = np.argpartition(d2, k - 1)[:k]
                nbr = obs[ii]

        if nbr.shape[0] < 8:
            c_local = nbr.mean(axis=0) if nbr.shape[0] > 0 else obs.mean(axis=0)
            return _normalize_cone((bottom - c_local).reshape(1, 3))[0]

        X = nbr - bottom[None, :]
        C = (X.T @ X) / float(max(1, X.shape[0]))
        _, evecs = np.linalg.eigh(C)
        u_escape = evecs[:, 0].astype(np.float64)
        u_escape = _normalize_cone(u_escape.reshape(1, 3))[0]

        c_local = nbr.mean(axis=0)
        out_u = _normalize_cone((bottom - c_local).reshape(1, 3))[0]
        if float(np.dot(u_escape, out_u)) < 0.0:
            u_escape = -u_escape
        return u_escape

    escape_u = _estimate_escape_direction()

    # ---------------------------
    # Build axis candidates inside a spherical cap around escape_u
    # ---------------------------
    dirs = _normalize_cone(_fibonacci_sphere_cone(int(n_dirs)))
    cap_cos = float(np.cos(np.deg2rad(float(CAP_ANGLE_DEG))))
    dots = (dirs @ escape_u.reshape(3, 1)).reshape(-1)
    cap_dirs = dirs[dots >= cap_cos]

    # Fallback if cap too sparse
    if cap_dirs.shape[0] < max(8, min(16, int(n_axes) // 4)):
        cap_cos2 = float(np.cos(np.deg2rad(80.0)))
        cap_dirs2 = dirs[(dirs @ escape_u.reshape(3, 1)).reshape(-1) >= cap_cos2]
        if cap_dirs2.shape[0] > cap_dirs.shape[0]:
            cap_dirs = cap_dirs2
    if cap_dirs.shape[0] == 0:
        cap_dirs = dirs

    if int(n_axes) < cap_dirs.shape[0]:
        axes = cap_dirs[rng.choice(cap_dirs.shape[0], size=int(n_axes), replace=False)]
    else:
        axes = cap_dirs

    # ---------------------------
    # Candidate obstacle subset for a frustum
    # ---------------------------
    def _candidate_obstacles_for_frustum(
        tip: np.ndarray,
        u: np.ndarray,
        tan_a: float,
        t0: float,
        t1: float,
        extra_pad: float,
    ) -> np.ndarray:
        if tree is None:
            return obs
        mid_t = 0.5 * (t0 + t1)
        mid = tip + u * mid_t
        half_len = 0.5 * (t1 - t0)
        rmax = float(t1) * float(tan_a)
        R = float(np.sqrt(half_len * half_len + (rmax + float(extra_pad)) ** 2))
        idx = tree.query_ball_point(mid, r=R)
        if not idx:
            return obs[:0]
        return obs[np.asarray(idx, dtype=np.int64)]

    # ---------------------------
    # Phase A obstruction test: min distance to ray segment (#3)
    # ---------------------------
    def _open_fraction_rays_segment(
        tip: np.ndarray,
        ray_dirs: np.ndarray,
        t0: float,
        t1: float,
        obs_subset: np.ndarray,
    ) -> float:
        ray_dirs = _normalize_cone(np.asarray(ray_dirs, dtype=np.float64).reshape(-1, 3))
        n_rays = int(ray_dirs.shape[0])
        if n_rays == 0:
            return 0.0
        if obs_subset.shape[0] == 0:
            return 1.0

        W = obs_subset.astype(np.float64, copy=False) - tip.reshape(1, 3)  # (K,3)
        proj_raw = W @ ray_dirs.T                                           # (K,R)
        proj_clamped = np.clip(proj_raw, float(t0), float(t1))              # (K,R)
        W2 = np.sum(W * W, axis=1, keepdims=True)                           # (K,1)
        dist2 = W2 - 2.0 * proj_clamped * proj_raw + proj_clamped * proj_clamped
        min_dist2 = np.min(dist2, axis=0)                                   # (R,)
        open_mask = min_dist2 >= float(r_clash_A) ** 2
        return float(np.mean(open_mask))

    # ---------------------------
    # Phase A sampling directions (precompute)
    # ---------------------------
    phis_surf_full = np.linspace(0.0, 2.0 * np.pi, int(N_SURFACE_FULL), endpoint=False).astype(np.float64)
    cphi_surf_full = np.cos(phis_surf_full)
    sphi_surf_full = np.sin(phis_surf_full)

    phis_surf_coarse = np.linspace(0.0, 2.0 * np.pi, int(N_SURFACE_COARSE), endpoint=False).astype(np.float64)
    cphi_surf_coarse = np.cos(phis_surf_coarse)
    sphi_surf_coarse = np.sin(phis_surf_coarse)

    u01_int = rng.uniform(0.0, 1.0, size=int(N_INTERIOR_FULL)).astype(np.float64)
    phi_int = rng.uniform(0.0, 2.0 * np.pi, size=int(N_INTERIOR_FULL)).astype(np.float64)
    cphi_int = np.cos(phi_int)
    sphi_int = np.sin(phi_int)

    # =========================
    # Phase A: choose (u, alpha) maximizing alpha
    # =========================
    best_alpha_deg = -np.inf
    best_u = None
    best_tip = None
    best_h0 = None

    for u_raw in axes:
        u = _normalize_cone(np.asarray(u_raw, dtype=np.float64).reshape(1, 3))[0]
        v, w = _orthonormal_basis_cone(u)

        if best_alpha_deg >= alpha_hi - 1e-9:
            break

        for a_deg in alpha_grid_deg:
            a_deg = float(a_deg)
            if a_deg <= best_alpha_deg + 1e-12:
                break

            open_req = float(np.clip(_open_req_for_alpha(a_deg), 0.0, 1.0))

            alpha = float(np.deg2rad(a_deg))
            tan_a = float(np.tan(alpha))
            if tan_a <= 1e-6:
                continue

            h0 = float(r_frustum_base_A / tan_a)
            tip = bottom - u * h0
            t_start = float(h0) + float(t_trunc)

            sin_a = float(np.sin(alpha))
            cos_a = float(np.cos(alpha))

            surface_dirs_coarse = (
                sin_a * (cphi_surf_coarse[:, None] * v[None, :] + sphi_surf_coarse[:, None] * w[None, :])
                + cos_a * u[None, :]
            )

            obs_cand_coarse = _candidate_obstacles_for_frustum(
                tip=tip, u=u, tan_a=tan_a,
                t0=t_start, t1=(t_start + float(RAY_LENGTH_A_COARSE)),
                extra_pad=float(r_clash_A),
            )

            surface_open_coarse = _open_fraction_rays_segment(
                tip=tip,
                ray_dirs=surface_dirs_coarse,
                t0=t_start,
                t1=(t_start + float(RAY_LENGTH_A_COARSE)),
                obs_subset=obs_cand_coarse,
            )

            if surface_open_coarse < open_req:
                continue

            need_full = (surface_open_coarse < (open_req + COARSE_MARGIN))

            if need_full:
                surface_dirs_full = (
                    sin_a * (cphi_surf_full[:, None] * v[None, :] + sphi_surf_full[:, None] * w[None, :])
                    + cos_a * u[None, :]
                )

                obs_cand_full = _candidate_obstacles_for_frustum(
                    tip=tip, u=u, tan_a=tan_a,
                    t0=t_start, t1=(t_start + float(RAY_LENGTH_A_FULL)),
                    extra_pad=float(r_clash_A),
                )

                surface_open_full = _open_fraction_rays_segment(
                    tip=tip,
                    ray_dirs=surface_dirs_full,
                    t0=t_start,
                    t1=(t_start + float(RAY_LENGTH_A_FULL)),
                    obs_subset=obs_cand_full,
                )
                if surface_open_full < open_req:
                    continue

                if int(N_INTERIOR_FULL) > 0:
                    z = cos_a + u01_int * (1.0 - cos_a)
                    rr = np.sqrt(np.maximum(0.0, 1.0 - z * z))
                    interior_dirs = (
                        rr[:, None] * (cphi_int[:, None] * v[None, :] + sphi_int[:, None] * w[None, :])
                        + z[:, None] * u[None, :]
                    )

                    interior_open_full = _open_fraction_rays_segment(
                        tip=tip,
                        ray_dirs=interior_dirs,
                        t0=t_start,
                        t1=(t_start + float(RAY_LENGTH_A_FULL)),
                        obs_subset=obs_cand_full,
                    )
                    if interior_open_full < open_req:
                        continue

            best_alpha_deg = a_deg
            best_u = u.astype(np.float64)
            best_tip = tip.astype(np.float64)
            best_h0 = float(h0)
            break

    if best_u is None:
        return (
            0.0,
            0.0,
            0.0,
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            0.0,
            False,
        )


    # =========================
    # Phase B: SOLVENT-ONLY height determination
    # =========================

    alpha = float(np.deg2rad(float(best_alpha_deg)))
    tan_a = float(np.tan(alpha))
    u = best_u
    tip = best_tip
    h0 = float(best_h0)

    step_A = float(step_A)

    # absolute heights from the bottom atom
    s_grid = np.arange(float(r_frustum_base_A), float(Dmax_A) + 1e-9, step_A)

    n_rim_B = 72
    phis_B = np.linspace(0.0, 2.0 * np.pi, n_rim_B, endpoint=False).astype(np.float64)
    cphi_B = np.cos(phis_B)
    sphi_B = np.sin(phis_B)

    v, w = _orthonormal_basis_cone(u)
    rim_dir = cphi_B[:, None] * v[None, :] + sphi_B[:, None] * w[None, :]

    t0_B = float(h0) + float(r_frustum_base_A)
    t1_B = float(h0) + float(Dmax_A)

    obs_cand_B = _candidate_obstacles_for_frustum(
        tip=tip, u=u, tan_a=tan_a,
        t0=t0_B, t1=t1_B,
        extra_pad=float(r_clash_A + r_solvent_shell_A),
    )

    def _nn_dist_to_subset(pts: np.ndarray, obs_subset: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        if obs_subset.shape[0] == 0:
            return np.full((pts.shape[0],), np.inf, dtype=np.float64)

        dif = obs_subset[None, :, :] - pts[:, None, :]
        d2 = np.sum(dif * dif, axis=2)
        return np.sqrt(np.min(d2, axis=1))

    D_cut = float(r_frustum_base_A)

    for s in s_grid:

        t = float(h0) + float(s)
        rc = t * tan_a
        center = tip + u * t

        r_outer = float(rc) + float(r_solvent_shell_A)
        outer_pts = center[None, :] + r_outer * rim_dir

        d_outer = _nn_dist_to_subset(outer_pts, obs_cand_B)
        outer_open = float(np.mean(d_outer >= float(r_clash_A)))

        if outer_open >= float(0.8):
            D_cut = float(s)
            break

        D_cut = float(s)

    height_capped = bool(D_cut >= float(Dmax_A))

    return (
        float(h0),
        float(D_cut),
        float(best_alpha_deg),
        tip.astype(np.float64),
        u.astype(np.float64),
        tip.astype(np.float64),
        float(h0 + D_cut),
        bool(height_capped),
    )


# ============================
# growability scoring
# ============================

    
def score_growability_atoms_tetrahedral(
    molH,
    df_atoms_with_sasa: pd.DataFrame,
    protein_coords: np.ndarray,
    probe_distance_A: float = 1.50,      # distance to first introduced atom
    r_excl_A: float = 3.20,              # local clash threshold for first atom
    r_frustum_base_A: float = 2.20,      # lower frustum radius for cone fitting
    r_clash_A: float = 2.0,
    r_solvent_shell_A: float = 1.75,
    sasa_eps_A2: float = 0.1,
    use_kdtree_if_available: bool = True,
    compute_cone: bool = True,
    cone_kwargs: dict | None = None,
    debug_print_cone: bool = False,
) -> pd.DataFrame:
    """
    Score growability per attachment heavy atom using ideal tetrahedral / trigonal / linear
    substituent directions.

    Non-cone outputs:
      - min_dist_to_protein_A
      - clash_margin_A
      - is_growable_local
      - growability_geom_status
      - growability_status
    """
    out = df_atoms_with_sasa.copy()
    cone_kwargs = {} if cone_kwargs is None else dict(cone_kwargs)

    if "sasa_A2" not in out.columns:
        raise KeyError("df_atoms_with_sasa must contain column 'sasa_A2' for SASA gating.")

    protein_coords = np.asarray(protein_coords, dtype=np.float64)
    has_obstacles = protein_coords.size > 0 and len(protein_coords) > 0

    tree = None
    if use_kdtree_if_available and has_obstacles:
        try:
            from scipy.spatial import cKDTree  
            tree = cKDTree(protein_coords)
        except Exception:
            tree = None

    conf = molH.GetConformer()

    best_min_dist = np.full(len(out), np.nan, dtype=np.float32)
    geom_statuses = np.array(["ok"] * len(out), dtype=object)

    cone_height = np.full(len(out), np.nan, dtype=np.float32)
    cone_height_clear = np.full(len(out), np.nan, dtype=np.float32)
    cone_height_capped = np.zeros(len(out), dtype=bool)
    cone_half_angle = np.full(len(out), np.nan, dtype=np.float32)
    cone_tip = np.full((len(out), 3), np.nan, dtype=np.float64)
    cone_dir = np.full((len(out), 3), np.nan, dtype=np.float64)
    cone_height_total = np.full(len(out), np.nan, dtype=np.float32)

    cone_status = np.array(["not_run"] * len(out), dtype=object)
    cone_reason = np.array([None] * len(out), dtype=object)
    cone_n_accept = np.zeros(len(out), dtype=np.int32)

    for i, row in enumerate(out.itertuples(index=False)):
        try:
            sasa = getattr(row, "sasa_A2", None)
            sasa_val = float(sasa) if sasa is not None and pd.notna(sasa) else 0.0
            if sasa_val <= float(sasa_eps_A2):
                geom_statuses[i] = "skipped_sasa"
                cone_status[i] = "skipped_sasa"
                cone_reason[i] = f"sasa_A2={sasa_val:.3f} <= {float(sasa_eps_A2):.3f}"
                continue

            attach_idx = int(getattr(row, "attach_atom_index"))

            Apos = conf.GetAtomPosition(attach_idx)
            A = np.array([Apos.x, Apos.y, Apos.z], dtype=np.float64)

            dirs = substituent_directions_for_atom(molH, conf, attach_idx)
            if not dirs:
                geom_statuses[i] = "failed"
                cone_status[i] = "cone_failed"
                cone_reason[i] = "no substituent directions for atom"
                if debug_print_cone:
                    print(f"[CONE-NO] attach_idx={attach_idx} | status=cone_failed | reason=no substituent directions")
                continue

            best = -1.0
            n_accept = 0
            best_cone_height_clear = np.inf
            best_cone_angle = -np.inf
            best_cone_result = None
            last_cone_error = None

            for u in dirs:
                u = np.asarray(u, dtype=np.float64)
                first_xyz = A + u * float(probe_distance_A)

                if not has_obstacles:
                    d = np.inf
                elif tree is not None:
                    d, _ = tree.query(first_xyz, k=1)
                    d = float(d)
                else:
                    d2 = np.sum((protein_coords - first_xyz) ** 2, axis=1)
                    d = float(np.sqrt(d2.min()))

                if d > best:
                    best = d

                if d >= float(r_excl_A):
                    n_accept += 1

                    if compute_cone:
                        try:
                            (
                                height_bottom,
                                height_clear,
                                ang_deg,
                                tip_xyz,
                                dir_u,
                                sphere_center_xyz,
                                sphere_radius_A,
                                height_capped,
                            ) = determine_dimension_cone(
                                obstacle_coords=protein_coords,
                                first_linker_atom_xyz=first_xyz,
                                r_frustum_base_A=float(r_frustum_base_A),
                                r_clash_A=float(r_clash_A),
                                r_solvent_shell_A=float(r_solvent_shell_A),
                                use_kdtree_if_available=use_kdtree_if_available,
                                **cone_kwargs,
                            )

                            if debug_print_cone:
                                print(
                                    f"cone: alpha_deg={float(ang_deg):.2f}, "
                                    f"height_clear={float(height_clear):.2f} Å, "
                                    f"height_bottom={float(height_bottom):.2f} Å, "
                                    f"capped={bool(height_capped)}"
                                )


                            h = float(height_clear)
                            a = float(ang_deg)

                            if np.isfinite(h) and np.isfinite(a):
                                candidate_is_zero = np.isclose(h, 0.0)
                                best_is_zero = np.isclose(best_cone_height_clear, 0.0)

                                if best_cone_result is None:
                                    take_candidate = True
                                elif best_is_zero and not candidate_is_zero:
                                    take_candidate = True
                                elif (not best_is_zero) and candidate_is_zero:
                                    take_candidate = False
                                elif h < best_cone_height_clear:
                                    take_candidate = True
                                elif np.isclose(h, best_cone_height_clear) and a > best_cone_angle:
                                    take_candidate = True
                                else:
                                    take_candidate = False

                                if take_candidate:
                                    best_cone_height_clear = h
                                    best_cone_angle = a
                                    best_cone_result = (
                                        float(height_bottom),
                                        float(height_clear),
                                        float(ang_deg),
                                        np.asarray(tip_xyz, dtype=np.float64).reshape(3,),
                                        np.asarray(dir_u, dtype=np.float64).reshape(3,),
                                        bool(height_capped),
                                    )

                        except Exception as e:
                            last_cone_error = f"{type(e).__name__}: {e}"
                            continue

            best_min_dist[i] = best
            cone_n_accept[i] = n_accept

            if not compute_cone:
                cone_status[i] = "not_run"
                cone_reason[i] = "compute_cone=False"
            elif n_accept == 0:
                cone_status[i] = "no_accept_dir"
                cone_reason[i] = f"no directions with d >= r_excl_A ({float(r_excl_A):.2f})"
            elif best_cone_result is None:
                if last_cone_error is not None:
                    cone_status[i] = "error"
                    cone_reason[i] = last_cone_error
                else:
                    cone_status[i] = "cone_failed"
                    cone_reason[i] = "determine_dimension_cone returned no finite (height,angle)"
            else:
                cone_status[i] = "ok"
                cone_reason[i] = None

                height_bottom, height_clear, ang_deg, tip_xyz, dir_u, height_capped = best_cone_result
                cone_height[i] = float(height_bottom)
                cone_height_clear[i] = float(height_clear)
                cone_height_total[i] = float(height_bottom + height_clear)
                cone_height_capped[i] = bool(height_capped)

                cone_half_angle[i] = float(ang_deg)
                cone_tip[i, :] = tip_xyz
                cone_dir[i, :] = dir_u

                if debug_print_cone:
                    print(
                        f"[CONE] attach_idx={attach_idx} | "
                        f"height_clear={height_clear:.3f} Å | half_angle={ang_deg:.2f}° | "
                        f"dir=({dir_u[0]:.3f},{dir_u[1]:.3f},{dir_u[2]:.3f}) | "
                        f"tip=({tip_xyz[0]:.3f},{tip_xyz[1]:.3f},{tip_xyz[2]:.3f}) | "
                        f"n_accept={n_accept}"
                    )

            if debug_print_cone and cone_status[i] != "ok":
                print(
                    f"[CONE-NO] attach_idx={attach_idx} | "
                    f"status={cone_status[i]} | "
                    f"reason={cone_reason[i]} | "
                    f"n_accept={n_accept} | "
                    f"best_min_dist={best:.3f}"
                )

        except Exception as e:
            geom_statuses[i] = "failed"
            cone_status[i] = "error"
            cone_reason[i] = f"{type(e).__name__}: {e}"

    out["probe_distance_A"] = float(probe_distance_A)
    out["r_excl_A"] = float(r_excl_A)
    out["r_frustum_base_A"] = float(r_frustum_base_A)
    out["min_dist_to_protein_A"] = best_min_dist
    out["clash_margin_A"] = out["min_dist_to_protein_A"] - float(r_excl_A)

    out["is_growable_local"] = (
        out["min_dist_to_protein_A"].ge(float(r_excl_A)) & out["min_dist_to_protein_A"].notna()
    )

    out["growability_geom_status"] = geom_statuses

    out["growability_status"] = (
        out["clash_margin_A"].gt(0)
        & out["sasa_A2"].gt(float(sasa_eps_A2))
        & out["min_dist_to_protein_A"].notna()
    )

    out["sasa_eps_A2"] = float(sasa_eps_A2)

    out["cone_height_A"] = cone_height
    out["cone_height_clear_A"] = cone_height_clear
    out["cone_height_total_A"] = cone_height_total
    out["cone_height_capped"] = cone_height_capped
    out["cone_half_angle_deg"] = cone_half_angle
    out["cone_tip_x"] = cone_tip[:, 0]
    out["cone_tip_y"] = cone_tip[:, 1]
    out["cone_tip_z"] = cone_tip[:, 2]
    out["cone_dir_x"] = cone_dir[:, 0]
    out["cone_dir_y"] = cone_dir[:, 1]
    out["cone_dir_z"] = cone_dir[:, 2]

    out["cone_status"] = cone_status
    out["cone_reason"] = cone_reason
    out["cone_n_accept_dirs"] = cone_n_accept

    return out


# ============================
# Topology descriptor
# ============================


def add_center_vs_end_descriptor(
    molH: Chem.Mol,
    df_atoms_with_sasa: pd.DataFrame,
    atom_index_col: str = "attach_atom_index",
    out_prefix: str = "topo_",
) -> pd.DataFrame:
    """
    Adds a simple ligand-intrinsic 'end vs center' descriptor based on
    normalized graph eccentricity computed on the heavy-atom graph.

    Outputs (added columns):
      - {out_prefix}eccentricity        (int)  max shortest-path distance to any heavy atom
      - {out_prefix}diameter            (int)  molecule heavy-atom graph diameter
      - {out_prefix}center_score        (float) 1 - ecc/diameter in [0,1] (diameter==0 -> 1.0)

    Notes:
      - molH may contain explicit H. Distances are computed on heavy atoms only.
      - df_atoms_with_sasa must have RDKit indices from molH in `atom_index_col`.
    """
    if atom_index_col not in df_atoms_with_sasa.columns:
        raise KeyError(f"df_atoms_with_sasa missing required column: {atom_index_col}")

    # Build mapping from molH heavy-atom indices -> heavy-only indices
    heavy_old_idxs = [a.GetIdx() for a in molH.GetAtoms() if a.GetAtomicNum() > 1]
    old_to_new = {old: new for new, old in enumerate(heavy_old_idxs)}

    # Heavy-only molecule (indices re-numbered, heavy atoms order preserved)
    mol_noH = Chem.RemoveHs(molH)

    n = mol_noH.GetNumAtoms()
    if n == 0:
        raise ValueError("Ligand has no heavy atoms after removing Hs.")

    # Heavy-atom topological distance matrix
    dist = rdmolops.GetDistanceMatrix(mol_noH).astype(int)

    # Eccentricity per heavy atom, and diameter for the ligand
    if n == 1:
        ecc = np.array([0], dtype=int)
        diameter = 0
    else:
        ecc = dist.max(axis=1)
        diameter = int(dist.max())

    # Center score in [0, 1]
    if diameter == 0:
        center_score = np.ones(n, dtype=float)  # single heavy atom: trivially "center"
    else:
        center_score = 1.0 - (ecc.astype(float) / float(diameter))

    # Map onto the dataframe rows (which reference molH atom indices)
    df_out = df_atoms_with_sasa.copy()
    old_idxs = df_out[atom_index_col].astype(int).to_numpy()

    new_idxs = np.array([old_to_new.get(i, -1) for i in old_idxs], dtype=int)
    if np.any(new_idxs < 0):
        missing = old_idxs[new_idxs < 0]
        raise ValueError(
            f"Some {atom_index_col} values are not heavy atoms in molH (or not present): "
            f"{sorted(set(missing.tolist()))[:20]}{'...' if len(set(missing.tolist())) > 20 else ''}"
        )

    df_out[f"{out_prefix}eccentricity"] = ecc[new_idxs]
    df_out[f"{out_prefix}diameter"] = diameter
    df_out[f"{out_prefix}center_score"] = center_score[new_idxs]

    return df_out


def merge_atom_growability_onto_sites(df_sites_with_sasa: pd.DataFrame,
                                     df_atoms_scored: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "attach_atom_index",
        "probe_distance_A",
        "min_dist_to_protein_A",
        "clash_margin_A",
        "is_growable_local",
        "growability_status",
    ]
    atom_small = df_atoms_scored[cols].copy()
    return df_sites_with_sasa.merge(atom_small, on="attach_atom_index", how="left")


# ============================
# pocket analysis
# ============================

        
def _get_xyz_columns(df: pd.DataFrame):
    """
    Return the names of the x/y/z coordinate columns using common conventions.
    """
    for cols in (("x", "y", "z"), ("X", "Y", "Z"), ("coord_x", "coord_y", "coord_z")):
        if all(c in df.columns for c in cols):
            return cols
    raise ValueError(f"Couldn't find xyz columns in df. Columns are: {list(df.columns)[:30]} ...")


def _fibonacci_sphere_pocket(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Even-ish directions on S^2. If rng is provided, apply a random rotation to de-alias;
    if rng is None, produce a deterministic set.

    Returns
    -------
    np.ndarray of shape (n, 3)
    """
    n = int(n)
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    i = np.arange(n, dtype=np.float64)
    phi = (1.0 + 5.0**0.5) / 2.0
    theta = 2.0 * np.pi * i / phi
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    dirs = np.stack([x, y, z], axis=1).astype(np.float64)

    if rng is None:
        return dirs

    # Random rotation using unit quaternion
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)  # w

    xq, yq, zq, wq = q1, q2, q3, q4
    R = np.array(
        [
            [1.0 - 2.0 * (yq * yq + zq * zq), 2.0 * (xq * yq - zq * wq),       2.0 * (xq * zq + yq * wq)],
            [2.0 * (xq * yq + zq * wq),       1.0 - 2.0 * (xq * xq + zq * zq), 2.0 * (yq * zq - xq * wq)],
            [2.0 * (xq * zq - yq * wq),       2.0 * (yq * zq + xq * wq),       1.0 - 2.0 * (xq * xq + yq * yq)],
        ],
        dtype=np.float64,
    )
    return (dirs @ R.T).astype(np.float64)


def _make_kdtree(coords: np.ndarray):
    """
    Return (tree, mode). If scipy is unavailable, returns (None, "none").
    """
    try:
        from scipy.spatial import cKDTree
        return cKDTree(coords), "scipy"
    except Exception:
        return None, "none"



    
def compute_pocket_df(
    molH,
    df_atoms: pd.DataFrame,
    protein_coords: np.ndarray,
    pdb_id: str | None = None,
    chosen=None,
    R_pocket: float = 5.0,
    R_contact: float = 4.0,
    n_rays: int = 200,
    ray_step: float = 0.75,
    ray_max: float = 15.0,
    block_R: float = 2.0,
) -> pd.DataFrame:
    """
    Return a 1-row dataframe with pocket descriptors around the ligand.

    Requires
    --------
    - df_atoms containing ligand heavy-atom indices in 'attach_atom_index'
    - protein_coords: (N, 3) numpy array of protein heavy-atom coordinates
    """
    if protein_coords is None or len(protein_coords) == 0:
        out = dict(
            pdb_id=pdb_id,
            pocket_n_atoms_R5=0,
            pocket_n_contacts_R4=0,
            pocket_enclosure=np.nan,
            pocket_open_fraction=np.nan,
            pocket_rg_A=np.nan,
        )
        return pd.DataFrame([out])

    if "attach_atom_index" not in df_atoms.columns:
        raise ValueError("df_atoms must contain attach_atom_index to map to molH")

    lig_atom_indices = df_atoms["attach_atom_index"].to_numpy(dtype=int)
    lig_xyz = ligand_coords_from_molH(molH, lig_atom_indices)

    if lig_xyz.shape[0] == 0:
        out = dict(
            pdb_id=pdb_id,
            pocket_n_atoms_R5=0,
            pocket_n_contacts_R4=0,
            pocket_enclosure=np.nan,
            pocket_open_fraction=np.nan,
            pocket_rg_A=np.nan,
        )
        if chosen is not None:
            out["chosen"] = str(chosen)
        return pd.DataFrame([out])

    center = lig_xyz.mean(axis=0)
    tree, mode = _make_kdtree(protein_coords)

    # (1) pocket_n_atoms_R5 + pocket atom coordinates (unique)
    if tree is not None:
        idx_lists = tree.query_ball_point(lig_xyz, r=R_pocket)
        pocket_idx = (
            np.unique(np.concatenate([np.asarray(ix, dtype=int) for ix in idx_lists if len(ix) > 0]))
            if any(len(ix) > 0 for ix in idx_lists)
            else np.array([], dtype=int)
        )
    else:
        pocket_mask = np.zeros(len(protein_coords), dtype=bool)
        for p in lig_xyz:
            d2 = np.sum((protein_coords - p) ** 2, axis=1)
            pocket_mask |= (d2 <= R_pocket**2)
        pocket_idx = np.where(pocket_mask)[0]

    pocket_n_atoms = int(pocket_idx.size)

    # (2) pocket_n_contacts_R4 (counts atom pairs)
    if tree is not None:
        contact_lists = tree.query_ball_point(lig_xyz, r=R_contact)
        pocket_n_contacts = int(sum(len(ix) for ix in contact_lists))
    else:
        pocket_n_contacts = 0
        for p in lig_xyz:
            d2 = np.sum((protein_coords - p) ** 2, axis=1)
            pocket_n_contacts += int(np.sum(d2 <= R_contact**2))

    # (3) pocket_rg_A: RMS spread of pocket atoms around ligand COM
    if pocket_n_atoms > 0:
        pocket_xyz = protein_coords[pocket_idx]
        pocket_rg = float(np.sqrt(np.mean(np.sum((pocket_xyz - center) ** 2, axis=1))))
    else:
        pocket_rg = np.nan

    # (4) pocket_enclosure: ray occlusion from ligand COM
    dirs = _fibonacci_sphere_pocket(n_rays)
    steps = np.arange(ray_step, ray_max + 1e-9, ray_step, dtype=float)
    blocked = 0

    if tree is not None:
        for d in dirs:
            hit = False
            for t in steps:
                p = center + t * d
                dist, _ = tree.query(p, k=1)
                if dist <= block_R:
                    hit = True
                    break
            blocked += int(hit)
    else:
        for d in dirs:
            hit = False
            for t in steps:
                p = center + t * d
                d2 = np.sum((protein_coords - p) ** 2, axis=1)
                if np.any(d2 <= block_R**2):
                    hit = True
                    break
            blocked += int(hit)

    pocket_enclosure = blocked / float(n_rays)
    pocket_open_fraction = 1.0 - pocket_enclosure

    out = dict(
        pdb_id=pdb_id,
        pocket_n_atoms_R5=pocket_n_atoms,
        pocket_n_contacts_R4=pocket_n_contacts,
        pocket_enclosure=float(pocket_enclosure),
        pocket_open_fraction=float(pocket_open_fraction),
        pocket_rg_A=float(pocket_rg) if np.isfinite(pocket_rg) else np.nan,
    )

    if chosen is not None:
        out["chosen"] = str(chosen)

    return pd.DataFrame([out])



# ============================
# complex summary and saving
# ============================


def compute_ligand_accessibility_summary_from_atoms(
    df_atoms: pd.DataFrame,
    accessible_col: str = "growability_status",
    center_col: str = "topo_center_score",
) -> dict:
    """
    Per-ligand (per PDB) summary metrics based on atoms.csv.

    - accessible_col: boolean criterion for linker attachment (growability_status)
    - center_col: position metric (topo_center_score)
    """
    if df_atoms is None or df_atoms.empty:
        return {
            "n_atoms_total": 0,
            "n_atoms_accessible": 0,
            "frac_atoms_accessible": np.nan,
            "min_center_score_accessible": np.nan,
            "max_center_score_accessible": np.nan,
            "min_center_score_all": np.nan,
            "max_center_score_all": np.nan,
        }

    n_total = len(df_atoms)

    # robust accessible boolean
    if accessible_col in df_atoms.columns:
        acc = df_atoms[accessible_col]
        if acc.dtype == object:
            acc_bool = acc.astype(str).str.lower().isin(["true", "1", "yes", "ok"])
        else:
            acc_bool = acc.fillna(False).astype(bool)
    else:
        acc_bool = pd.Series([False] * n_total, index=df_atoms.index)

    n_acc = int(acc_bool.sum())
    frac_acc = (n_acc / n_total) if n_total > 0 else np.nan

    # convert center score column
    center_all = (
        pd.to_numeric(df_atoms[center_col], errors="coerce")
        if center_col in df_atoms.columns
        else pd.Series([np.nan] * n_total, index=df_atoms.index)
    )

    # extrema over ALL atoms
    if center_all.notna().any():
        min_all = float(center_all.min())
        max_all = float(center_all.max())
    else:
        min_all = np.nan
        max_all = np.nan

    # extrema over ACCESSIBLE atoms only
    if n_acc > 0 and center_col in df_atoms.columns:
        center_acc = center_all.loc[acc_bool]
        if center_acc.notna().any():
            min_acc = float(center_acc.min())
            max_acc = float(center_acc.max())
        else:
            min_acc = np.nan
            max_acc = np.nan
    else:
        min_acc = np.nan
        max_acc = np.nan

    return {
        "n_atoms_total": int(n_total),
        "n_atoms_accessible": int(n_acc),
        "frac_atoms_accessible": float(frac_acc) if frac_acc == frac_acc else np.nan,
        "min_center_score_accessible": min_acc,
        "max_center_score_accessible": max_acc,
        "min_center_score_all": min_all,
        "max_center_score_all": max_all,
    }

def coerce_single_row_pocket(df_pocket: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """
    Ensure df_pocket is a 1-row dataframe.

    Returns:
      (df_pocket_1row or None, warning_message or None)
    """
    if df_pocket is None:
        return None, "df_pocket is None"
    if df_pocket.empty:
        return None, "df_pocket is empty"

    if df_pocket.shape[0] == 1:
        return df_pocket, None

    # If multiple rows, take the first row but warn
    warn = f"df_pocket had {df_pocket.shape[0]} rows; using first row for complex.csv"
    return df_pocket.iloc[[0]].copy(), warn


def build_df_complex_from_atoms_and_pocket(
    df_atoms: pd.DataFrame,
    df_pocket: pd.DataFrame,
    accessible_col: str = "growability_status",
    center_col: str = "topo_center_score",
) -> pd.DataFrame:
    if df_pocket is None or df_pocket.empty:
        raise ValueError("df_pocket is empty; expected a single-row pocket dataframe.")
    if df_pocket.shape[0] != 1:
        raise ValueError(f"Expected df_pocket to have 1 row, got {df_pocket.shape[0]}.")

    # --- ligand-level summary ---
    summary = compute_ligand_accessibility_summary_from_atoms(
        df_atoms=df_atoms,
        accessible_col=accessible_col,
        center_col=center_col,
    )

    # --- ensure ALL-atoms min/max center score are present ---
    if df_atoms is not None and not df_atoms.empty and center_col in df_atoms.columns:
        center_all = pd.to_numeric(df_atoms[center_col], errors="coerce")
        if center_all.notna().any():
            min_all = float(center_all.min())
            max_all = float(center_all.max())
        else:
            min_all = np.nan
            max_all = np.nan
    else:
        min_all = np.nan
        max_all = np.nan

    summary.setdefault("min_center_score_all", min_all)
    summary.setdefault("max_center_score_all", max_all)

    df_complex = df_pocket.copy()
    for k, v in summary.items():
        df_complex[k] = v

    return df_complex


def save_structure_result_csv(
    base_dir: str | Path,
    pdb_id: str,
    chosen: dict,
    molH,
    df_atoms: pd.DataFrame,
    df_sites: pd.DataFrame,
    df_pocket: pd.DataFrame,
    params: dict,
    status: str = "ok",
    error: str | None = None,
) -> pd.DataFrame | None:
    """
    Save per-structure outputs into base_dir/pdb_id.

    Writes:
      - atoms.csv
      - sites.csv
      - pocket.csv
      - complex.csv  (pocket + ligand summary)  [best effort]
      - meta.json

    Returns:
      - df_complex (1-row DataFrame) if created successfully, else None
    """
    base_dir = Path(base_dir)
    outdir = base_dir / pdb_id
    outdir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Core CSV outputs
    # ----------------------------
    if df_atoms is not None:
        df_atoms.to_csv(outdir / "atoms.csv", index=False)

    if df_sites is not None:
        df_sites.to_csv(outdir / "sites.csv", index=False)

    if df_pocket is not None:
        df_pocket.to_csv(outdir / "pocket.csv", index=False)

    # ----------------------------
    # complex.csv (best effort)
    # ----------------------------
    df_complex = None
    complex_warning_msgs = []

    if status == "ok":
        try:
            df_pocket_1row, pocket_warn = coerce_single_row_pocket(df_pocket)
            if pocket_warn:
                complex_warning_msgs.append(pocket_warn)

            if df_pocket_1row is None:
                raise ValueError("Cannot build complex.csv because df_pocket is missing/empty.")

            df_complex = build_df_complex_from_atoms_and_pocket(
                df_atoms=df_atoms,
                df_pocket=df_pocket_1row,            
                accessible_col="growability_status",
                center_col="topo_center_score",
            )
            df_complex.to_csv(outdir / "complex.csv", index=False)

        except Exception as e:
            complex_warning_msgs.append(f"Failed to write complex.csv: {repr(e)}")
            df_complex = None

        if complex_warning_msgs:
            (outdir / "complex_warning.txt").write_text("\n".join(complex_warning_msgs) + "\n")

    # ----------------------------
    # ligand.sdf (RDKit molecule)
    # ----------------------------
    ligand_path = outdir / "ligand.sdf"
    try:
        write_ligand_sdf_safe(molH, ligand_path)
    except Exception as e:
        # Remove empty/partial file if it was created
        try:
            if ligand_path.exists() and ligand_path.stat().st_size == 0:
                ligand_path.unlink()
        except Exception:
            pass

        (outdir / "ligand_warning.txt").write_text(
            f"Failed to write ligand.sdf: {repr(e)}\n"
        )

    # ----------------------------
    # meta.json (json-safe)
    # ----------------------------
    meta = {
        "pdb_id": pdb_id,
        "status": status,
        "error": error,
        "chosen": make_json_safe(chosen),
        "params": make_json_safe(params),
        "timestamp_unix": time.time(),
        "n_atoms_rows": int(0 if df_atoms is None else len(df_atoms)),
        "n_sites_rows": int(0 if df_sites is None else len(df_sites)),
        "n_pocket_rows": int(0 if df_pocket is None else len(df_pocket)),
        "has_complex_csv": bool(df_complex is not None),
    }

    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    return df_complex



# -----------------------------
# 3) Transactional wrapper
#    (call THIS from your batch loop)
# -----------------------------
def save_structure_result_csv_transactional(
    base_dir: str | Path,
    pdb_id: str,
    chosen: dict,
    molH,
    df_atoms: pd.DataFrame,
    df_sites: pd.DataFrame,
    df_pocket: pd.DataFrame,
    params: dict,
) -> pd.DataFrame | None:
    """
    Transactional saver:
      - Writes outputs into a temporary folder first
      - Atomically renames temp -> base_dir/pdb_id only if everything succeeded
      - Ensures NO output folder exists for failures/partial writes

    Returns:
      - df_complex (1-row DataFrame) if created successfully, else None
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Guard: prevent accidental recursive transactional saving
    if str(pdb_id).startswith(".tmp_"):
        raise ValueError(
            f"Refusing transactional save with pdb_id={pdb_id!r} (looks like a temp id). "
            "This indicates the transactional wrapper is being called recursively."
        )

    final_dir = base_dir / pdb_id
    tmp_dir = base_dir / f".tmp_{pdb_id}_{int(time.time() * 1000)}"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    try:
        df_complex = save_structure_result_csv(
            base_dir=base_dir,
            pdb_id=tmp_dir.name,   
            chosen=chosen,
            molH=molH,
            df_atoms=df_atoms,
            df_sites=df_sites,
            df_pocket=df_pocket,
            params=params,
            status="ok",
            error=None,
        )

        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)

        tmp_dir.rename(final_dir)

        return df_complex

    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise



# ============================
# batch execution
# ============================

def run_batch(
    pdb_ids,
    outdir: str | Path,
    params: dict,
    verbose: bool = False,
    *,
    compute_cone: bool = False,
    debug_print_cone: bool = False,
    cone_kwargs: dict | None = None,
) -> pd.DataFrame:

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    complex_rows = []

    print(f"\n=== Batch run started ({time.strftime('%Y-%m-%dT%H:%M:%S')}) ===")
    print(f"Output directory: {outdir}")
    print(f"Number of PDB IDs: {len(pdb_ids)}")
    print("Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    if cone_kwargs:
        print("Cone kwargs:")
        for k, v in cone_kwargs.items():
            print(f"  {k}: {v}")
    print()

    log_rows = []

    for i, pid in enumerate(pdb_ids, start=1):
        t0 = time.time()
        print(f"[{i}/{len(pdb_ids)}] {pid}: starting...")

        try:
            res = analyze_pdb_ligand_exposure(pid, verbose=verbose)
            if res.get("status") != "ok":
                raise RuntimeError(
                    res.get("reason", "analyze_pdb_ligand_exposure returned status != ok")
                )

            molH = res["molH"]
            chosen = res["chosen"]
            structure_clean = res["structure_clean"]
            df_atoms = res["atoms"]
            df_sites = res["sites"]

            protein_coords, _ = get_obstacle_heavy_atom_coords(
                structure_clean,
                chosen=chosen,
                exclude_waters=params.get("exclude_waters", True),
                exclude_ligand_instance=params.get("exclude_ligand_instance", True),
                keep_only_polymer=params.get("keep_only_polymer", True),
            )

            df_atoms_scored = score_growability_atoms_tetrahedral(
                molH=molH,
                df_atoms_with_sasa=df_atoms,
                protein_coords=protein_coords,
                probe_distance_A=params.get("probe_distance_A", 1.5),
                r_excl_A=params.get("r_excl_A", 3.2),
                r_frustum_base_A=params.get("r_frustum_base_A", 2.2),
                r_clash_A=params.get("r_clash_A", 1.4),
                r_solvent_shell_A=params.get("r_solvent_shell_A", 1.75),
                sasa_eps_A2=params.get("sasa_eps_A2", 0.1),
                use_kdtree_if_available=params.get("use_kdtree_if_available", True),
                compute_cone=compute_cone,
                cone_kwargs=cone_kwargs,
                debug_print_cone=debug_print_cone,
            )

            df_atoms_scored = add_center_vs_end_descriptor(
                molH=molH,
                df_atoms_with_sasa=df_atoms_scored,
            )

            df_sites_scored = merge_atom_growability_onto_sites(df_sites, df_atoms_scored)

            required_cols = ["growability_status", "topo_center_score"]
            missing = [c for c in required_cols if c not in df_atoms_scored.columns]
            if missing:
                raise RuntimeError(
                    f"Refusing to save: df_atoms_scored missing required columns {missing}. "
                    f"Available cols include: {list(df_atoms_scored.columns)[:25]} ..."
                )

            df_pocket = compute_pocket_df(
                df_atoms=df_atoms_scored,
                protein_coords=protein_coords,
                pdb_id=pid,
                chosen=chosen,
                molH=molH,
            )

            df_complex = save_structure_result_csv_transactional(
                base_dir=outdir,
                pdb_id=pid,
                chosen=chosen,
                molH=molH,
                df_atoms=df_atoms_scored,
                df_sites=df_sites_scored,
                df_pocket=df_pocket,
                params=params,
            )

            if df_complex is not None and not df_complex.empty:
                complex_rows.append(df_complex)

            secs = time.time() - t0
            print(f"[{i}/{len(pdb_ids)}] {pid}: OK ({secs:.3f} s)")

            log_rows.append({
                "pdb_id": pid,
                "status": "ok",
                "error": None,
                "seconds": secs,
                "outdir": str(outdir / pid),
            })

        except Exception as e:
            secs = time.time() - t0
            print(f"[{i}/{len(pdb_ids)}] {pid}: FAILED ({secs:.3f} s)")
            print(f"    -> {type(e).__name__}: {e}")

            log_rows.append({
                "pdb_id": pid,
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
                "seconds": secs,
                "outdir": str(outdir / pid),
            })

    df_complex_all = pd.concat(complex_rows, ignore_index=True) if complex_rows else pd.DataFrame()
    complex_all_path = outdir / "complex_all.csv"
    df_complex_all.to_csv(complex_all_path, index=False)

    log_df = pd.DataFrame(log_rows)
    (outdir / "batch_log.csv").write_text(log_df.to_csv(index=False))
    (outdir / "batch_log.txt").write_text(log_df.to_string(index=False))

    print("\n=== Batch run finished ===")
    print(f"Saved: {outdir / 'batch_log.csv'}")
    print(f"Saved: {outdir / 'batch_log.txt'}")
    print(f"Saved: {complex_all_path}")

    return log_df


# ============================
# aggregation utilities
# ============================


def save_all_atoms_one_file(
    base_dir: str | Path,
    out_csv: str | Path | None = None,
    atoms_filename: str = "atoms.csv",
    include_failed: bool = False,
) -> pd.DataFrame:
    """
    Aggregate per-PDB atoms.csv into a single CSV file.

    Assumes per-PDB structure:
      base_dir/<PDB_ID>/atoms.csv

    Writes:
      base_dir/all_atoms.csv   (default)

    Two-pass design:
      Pass 1: read only headers to compute union of columns across files
      Pass 2: stream append each file reindexed to union columns

    Adds:
      - pdb_id column derived from the subfolder name (base_dir/<PDB_ID>/...)

    Parameters
    ----------
    base_dir : str | Path
        Base output directory containing per-PDB subfolders.
    out_csv : str | Path | None
        Output CSV path. Default: base_dir / "all_atoms.csv"
    atoms_filename : str
        Name of the per-PDB atoms file (default "atoms.csv")
    include_failed : bool
        If True, also include any atoms.csv found anywhere under base_dir,
        even if folder naming isn't a strict PDB ID. Usually keep False.

    Returns
    -------
    pd.DataFrame
        Log table with one row per attempted file: pdb_id, path, n_rows, status, error.
    """
    base_dir = Path(base_dir)
    if out_csv is None:
        out_csv = base_dir / "all_atoms.csv"
    else:
        out_csv = Path(out_csv)

    pdb_col = "pdb_id"

    if include_failed:
        atom_paths = sorted(base_dir.rglob(atoms_filename))
    else:
        atom_paths = sorted(p for p in base_dir.glob(f"*/{atoms_filename}") if p.is_file())

    if not atom_paths:
        raise FileNotFoundError(f"No '{atoms_filename}' files found under: {base_dir}")

    # ---------- Pass 1: union of columns ----------
    col_union: list[str] = []
    col_set = set()

    col_set.add(pdb_col)
    col_union.append(pdb_col)

    scan_log = []
    for p in atom_paths:
        pdb_id = p.parent.name
        try:
            cols = list(pd.read_csv(p, nrows=0).columns)
            for c in cols:
                if c not in col_set:
                    col_set.add(c)
                    col_union.append(c)
            scan_log.append({"pdb_id": pdb_id, "path": str(p), "status": "ok", "error": None})
        except Exception as e:
            scan_log.append(
                {"pdb_id": pdb_id, "path": str(p), "status": "failed", "error": f"{type(e).__name__}: {e}"}
            )

    readable_paths = [Path(r["path"]) for r in scan_log if r["status"] == "ok"]

    if not readable_paths:
        raise RuntimeError("All atoms.csv files failed to read headers; cannot aggregate.")

    # ---------- Pass 2: stream-write ----------
    if out_csv.exists():
        out_csv.unlink()

    write_log = []
    wrote_header = False

    for p in readable_paths:
        pdb_id = p.parent.name
        try:
            df = pd.read_csv(p)

            # Always set/overwrite pdb_id from folder name
            df[pdb_col] = pdb_id

            # Reindex to union columns (missing columns become NaN)
            df = df.reindex(columns=col_union)

            # Append to output
            df.to_csv(out_csv, mode="a", index=False, header=(not wrote_header))
            wrote_header = True

            write_log.append({"pdb_id": pdb_id, "path": str(p), "n_rows": int(len(df)), "status": "ok", "error": None})
        except Exception as e:
            write_log.append(
                {"pdb_id": pdb_id, "path": str(p), "n_rows": None, "status": "failed", "error": f"{type(e).__name__}: {e}"}
            )

    df_write_log = pd.DataFrame(write_log)

    agg_log_csv = out_csv.with_name(out_csv.stem + "_log.csv")
    df_write_log.to_csv(agg_log_csv, index=False)

    print(f"Saved aggregated atoms to: {out_csv}")
    print(f"Saved aggregation log to: {agg_log_csv}")
    print(f"Files aggregated successfully: {(df_write_log['status'] == 'ok').sum()} / {len(df_write_log)}")
    print(f"Total rows written: {df_write_log.loc[df_write_log['status']=='ok', 'n_rows'].sum()}")

    return df_write_log

    
def save_all_complex_one_file(root_folder_path: str | Path) -> pd.DataFrame | None:
    """
    Aggregate all complex.csv files found under root_folder_path into one file.

    Saves:
      root_folder_path / "aggregated_complex.csv"
    """
    root_folder = Path(root_folder_path)

    csv_files = list(root_folder.rglob("complex.csv"))
    dataframes = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df["source_folder"] = file.parent.name
            dataframes.append(df)
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")

    if not dataframes:
        print("No complex.csv files found.")
        return None

    combined_df = pd.concat(dataframes, ignore_index=True)

    output_path = root_folder / "aggregated_complex.csv"
    combined_df.to_csv(output_path, index=False)

    print(f"Aggregated file saved to: {output_path}")
    return combined_df


import pandas as pd
import numpy as np
from pathlib import Path



# ============================
# cone drawing
# ============================



def _orthonormal_basis_draw(u: np.ndarray):
    """Return (u_hat, v_hat, w_hat) with v,w perpendicular to u."""
    u = np.asarray(u, dtype=float).reshape(3,)
    nu = np.linalg.norm(u)
    if nu < 1e-12:
        raise ValueError("axis_u has near-zero norm")
    u = u / nu

    tmp = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(tmp, u))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=float)

    v = np.cross(u, tmp)
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        tmp = np.array([0.0, 0.0, 1.0], dtype=float)
        v = np.cross(u, tmp)
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            raise ValueError("Failed to build orthonormal basis")
    v /= nv
    w = np.cross(u, v)
    return u, v, w


def _format_pdb_hetatm(atom_id: int, atom_name: str, resn: str, chain: str, resi: int, xyz, element: str):
    x, y, z = xyz
    return (
        f"HETATM{atom_id:5d} {atom_name:<4s}{resn:>4s} {chain:1s}{resi:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00  0.00           {element:>2s}"
    )


def _sphere_points(center_xyz, radius_A, n_lat=10, n_lon=20):
    """
    Points on a sphere surface using latitude/longitude sampling.
    n_lat: number of latitude bands between poles (excluding poles)
    n_lon: points around longitude for each band
    """
    c = np.asarray(center_xyz, dtype=float).reshape(3,)
    pts = []

    pts.append(c + np.array([0.0, 0.0, radius_A]))

    for i in range(1, n_lat + 1):
        phi = np.pi * i / (n_lat + 1)  # (0, pi)
        z = radius_A * np.cos(phi)
        r_xy = radius_A * np.sin(phi)
        for j in range(n_lon):
            theta = 2.0 * np.pi * j / n_lon
            x = r_xy * np.cos(theta)
            y = r_xy * np.sin(theta)
            pts.append(c + np.array([x, y, z]))

    pts.append(c + np.array([0.0, 0.0, -radius_A]))

    return pts


def write_frustum_with_origin_sphere_as_pdb(
    tip_xyz,
    axis_u,
    half_angle_deg,
    out_path,
    s0_A,                 # start distance from tip (lower cut)
    s1_A,                 # end distance from tip (upper cut)
    n_height_steps=30,
    n_angular_steps=72,
    *,
    sphere_radius_A=1.0,  # "small sphere" size
    sphere_n_lat=10,
    sphere_n_lon=20,
):
    """
    Writes:
      1) The frustum surface between s0 and s1 (measured from tip_xyz along axis_u)
      2) A small sphere centered at the frustum lower-plane center (start point)

    Transparency is NOT stored in PDB; instead the sphere is placed in residue SPH
    so you can make it transparent in your viewer.
    """
    if s1_A <= s0_A:
        raise ValueError("s1_A must be > s0_A")

    tip = np.asarray(tip_xyz, dtype=float).reshape(3,)
    u_hat, v_hat, w_hat = _orthonormal_basis(axis_u)

    alpha = np.deg2rad(float(half_angle_deg))
    tan_a = float(np.tan(alpha))

    # Frustum geometry derived from truncation distances
    start_xyz = tip + u_hat * float(s0_A)
    start_radius_A = float(s0_A) * tan_a
    height_A = float(s1_A) - float(s0_A)

    atom_lines = []
    atom_id = 1

    # ---- (A) Sphere at lower-plane center ----
    # Put sphere atoms in residue SPH, chain B, so they are easy to select
    sph_pts = _sphere_points(start_xyz, float(sphere_radius_A), n_lat=sphere_n_lat, n_lon=sphere_n_lon)
    for p in sph_pts:
        atom_lines.append(_format_pdb_hetatm(atom_id, "S", "SPH", "B", 1, p, "S"))
        atom_id += 1

    # ---- (B) Frustum surface ----
    # Put frustum atoms in residue CON, chain A
    for i in range(n_height_steps + 1):
        s = height_A * (i / n_height_steps)        # distance from start_xyz
        radius = start_radius_A + s * tan_a
        center = start_xyz + u_hat * s

        for j in range(n_angular_steps):
            theta = 2.0 * np.pi * j / n_angular_steps
            pt = center + radius * (np.cos(theta) * v_hat + np.sin(theta) * w_hat)
            atom_lines.append(_format_pdb_hetatm(atom_id, "C", "CON", "A", 1, pt, "C"))
            atom_id += 1

    atom_lines.append("END")
    Path(out_path).write_text("\n".join(atom_lines))

def draw_cone_from_atoms_csv(
    atoms_csv_path: str | Path,
    out_pdb_path: str | Path,
    *,
    row_index: int | None = None,
    attach_atom_index: int | None = None,
):
    atoms_csv_path = Path(atoms_csv_path)
    out_pdb_path = Path(out_pdb_path)

    df = pd.read_csv(atoms_csv_path)

    if attach_atom_index is not None:
        sel = df[df["attach_atom_index"] == int(attach_atom_index)]
        if sel.empty:
            raise ValueError(f"No row with attach_atom_index={attach_atom_index}")
        row = sel.iloc[0]
    elif row_index is not None:
        row = df.iloc[int(row_index)]
    else:
        raise ValueError("Provide either row_index or attach_atom_index")

    def pick(row, *names):
        for n in names:
            if n in row.index and pd.notna(row[n]):
                return row[n]
        raise KeyError(f"None of these columns exist / are valid: {names}")

    tip_xyz = [pick(row, "cone_tip_x"), pick(row, "cone_tip_y"), pick(row, "cone_tip_z")]
    axis_u  = [pick(row, "cone_dir_x"), pick(row, "cone_dir_y"), pick(row, "cone_dir_z")]
    alpha   = float(pick(row, "cone_half_angle_deg", "cone_half_angle"))

    h_bottom = float(pick(row, "cone_height_A", "cone_height_bottom_A"))
    h_clear  = float(pick(row, "cone_height_clear_A", "cone_height_clear"))

    write_frustum_with_origin_sphere_as_pdb(
        tip_xyz=tip_xyz,
        axis_u=axis_u,
        half_angle_deg=alpha,
        out_path=out_pdb_path,
        s0_A=h_bottom,
        s1_A=h_bottom + h_clear,
        sphere_radius_A=2.2,
    )

    print(f"Wrote: {out_pdb_path}")

    

# ============================
# Entry point
# ============================

def main():
    pdb_ids = ["9G7H"]    # Replace by PDBs

    print(pdb_ids)

    # Output directory (TODO: replace with your desired path)
    outdir = Path("PATH/TO/OUTPUT_DIRECTORY")

    # Global analysis parameters
    params = {
        "n_dirs": 96,
        "probe_distance_A": 1.5,      # distance to first introduced atom
        "r_excl_A": 3.2,              # local clash threshold for first atom
        "r_frustum_base_A": 2.2,      # lower radius of frustum
        "r_clash_A": 2.0,             # bead radius for cone clash checks
        "r_solvent_shell_A": 1.75,    # solvent shell at upper frustum
        "exclude_waters": True,
        "exclude_ligand_instance": True,
        "keep_only_polymer": True,
    }

    # Run batch analysis
    log_df = run_batch(
        pdb_ids=pdb_ids,
        outdir=outdir,
        params=params,
        compute_cone=True,
        debug_print_cone=True,
    )

    # Summary
    print("\nBatch log summary:")
    print(log_df)


# ============================
# Main code
# ============================

if __name__ == "__main__":
    RUN_BATCH = True
    DRAW_CONE = False

    if RUN_BATCH:
        main()

    if DRAW_CONE:
        draw_cone_from_atoms_csv(
            # TODO: replace with your input file path
            atoms_csv_path=Path("PATH/TO/atoms.csv"),

            # TODO: replace with your desired output file path
            out_pdb_path=Path("PATH/TO/output_cone.pdb"),

            attach_atom_index=2,
        )














    




 










   