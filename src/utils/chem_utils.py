"""
化学工具模块 - 提供分子处理和验证功能
"""
from typing import Optional, List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, SaltRemover, MolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np


class MoleculeProcessor:
    """分子处理器"""
    
    def __init__(self):
        """初始化分子处理器"""
        self.salt_remover = SaltRemover.SaltRemover()
        self.uncharger = MolStandardize.rdMolStandardize.Uncharger()
    
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        将SMILES转换为RDKit分子对象
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            RDKit分子对象，如果转换失败返回None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except:
            return None
    
    def mol_to_smiles(self, mol: Chem.Mol, canonical: bool = True) -> Optional[str]:
        """
        将RDKit分子对象转换为SMILES
        
        Args:
            mol: RDKit分子对象
            canonical: 是否使用规范SMILES
            
        Returns:
            SMILES字符串
        """
        try:
            if canonical:
                return Chem.MolToSmiles(mol)
            else:
                return Chem.MolToSmiles(mol, canonical=False)
        except:
            return None
    
    def standardize_mol(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        标准化分子
        - 去盐
        - 中性化
        - 标准化
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            标准化后的分子对象
        """
        if mol is None:
            return None
        
        try:
            # 去盐
            mol = self.salt_remover.StripMol(mol)
            
            # 中性化
            mol = self.uncharger.uncharge(mol)
            
            # 标准化（更新分子）
            Chem.SanitizeMol(mol)
            
            return mol
        except:
            return None
    
    def process_smiles(self, smiles: str) -> Tuple[Optional[str], Optional[Chem.Mol]]:
        """
        处理SMILES字符串：转换→标准化→规范化
        
        Args:
            smiles: 输入的SMILES字符串
            
        Returns:
            (标准化的SMILES, 分子对象)
        """
        # 转换为分子对象
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None, None
        
        # 标准化
        mol = self.standardize_mol(mol)
        if mol is None:
            return None, None
        
        # 获取规范SMILES
        canonical_smiles = self.mol_to_smiles(mol, canonical=True)
        
        return canonical_smiles, mol


class MoleculeValidator:
    """分子验证器"""
    
    def __init__(
        self,
        mw_range: Tuple[float, float] = (150, 800),
        heavy_atom_range: Tuple[int, int] = (10, 60)
    ):
        """
        初始化验证器
        
        Args:
            mw_range: 分子量范围
            heavy_atom_range: 重原子数范围
        """
        self.mw_min, self.mw_max = mw_range
        self.ha_min, self.ha_max = heavy_atom_range
    
    def is_valid(self, mol: Chem.Mol) -> bool:
        """
        检查分子是否有效
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            是否有效
        """
        if mol is None:
            return False
        
        # 检查分子量
        mw = Descriptors.MolWt(mol)
        if not (self.mw_min <= mw <= self.mw_max):
            return False
        
        # 检查重原子数
        heavy_atoms = mol.GetNumHeavyAtoms()
        if not (self.ha_min <= heavy_atoms <= self.ha_max):
            return False
        
        # 检查是否包含金属
        if self.contains_metal(mol):
            return False
        
        return True
    
    def contains_metal(self, mol: Chem.Mol) -> bool:
        """
        检查分子是否包含金属元素
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            是否包含金属
        """
        metals = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',  # 碱金属
                  'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',  # 碱土金属
                  'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Pd', 'Pt', 'Au', 'Ag',  # 过渡金属（常见）
                  'Al', 'Ga', 'In', 'Sn', 'Pb', 'Bi']  # 其他金属
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in metals:
                return True
        return False
    
    def get_validation_info(self, mol: Chem.Mol) -> dict:
        """
        获取分子验证信息
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            验证信息字典
        """
        if mol is None:
            return {"valid": False, "reason": "Invalid molecule"}
        
        info = {
            "valid": True,
            "molecular_weight": Descriptors.MolWt(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "contains_metal": self.contains_metal(mol),
            "reasons": []
        }
        
        # 检查各项指标
        if not (self.mw_min <= info["molecular_weight"] <= self.mw_max):
            info["valid"] = False
            info["reasons"].append(
                f"分子量超出范围 ({self.mw_min}-{self.mw_max})"
            )
        
        if not (self.ha_min <= info["heavy_atoms"] <= self.ha_max):
            info["valid"] = False
            info["reasons"].append(
                f"重原子数超出范围 ({self.ha_min}-{self.ha_max})"
            )
        
        if info["contains_metal"]:
            info["valid"] = False
            info["reasons"].append("包含金属元素")
        
        return info


def get_murcko_scaffold(mol: Chem.Mol) -> Optional[str]:
    """
    获取分子的Bemis-Murcko骨架
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        骨架SMILES
    """
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None


def generate_conformer(
    mol: Chem.Mol,
    num_conformers: int = 1,
    optimize: bool = True
) -> Optional[Chem.Mol]:
    """
    为分子生成3D构象
    
    Args:
        mol: RDKit分子对象
        num_conformers: 生成的构象数量
        optimize: 是否使用力场优化
        
    Returns:
        带有3D构象的分子对象
    """
    if mol is None:
        return None
    
    try:
        # 添加氢
        mol = Chem.AddHs(mol)
        
        # 生成构象
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_conformers,
            params=AllChem.ETKDG()
        )
        
        # 优化（如果需要）
        if optimize and mol.GetNumConformers() > 0:
            for conf_id in range(mol.GetNumConformers()):
                AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        
        return mol
    except:
        return None


def calculate_descriptors(mol: Chem.Mol) -> dict:
    """
    计算分子描述符
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        描述符字典
    """
    if mol is None:
        return {}
    
    descriptors = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'RotatableBonds': Descriptors.NumRotatableBonds(mol),
        'AromaticRings': Descriptors.NumAromaticRings(mol),
        'HeavyAtoms': mol.GetNumHeavyAtoms(),
    }
    
    return descriptors


def get_inchi_key(mol: Chem.Mol) -> Optional[str]:
    """
    获取分子的InChI Key（用于去重）
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        InChI Key字符串
    """
    try:
        return Chem.MolToInchiKey(mol)
    except:
        return None


if __name__ == "__main__":
    # 测试化学工具
    print("测试化学工具...")
    
    # 测试分子处理
    processor = MoleculeProcessor()
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # 阿司匹林
    
    print(f"\n原始SMILES: {test_smiles}")
    canonical_smiles, mol = processor.process_smiles(test_smiles)
    print(f"标准化SMILES: {canonical_smiles}")
    
    # 测试分子验证
    validator = MoleculeValidator()
    is_valid = validator.is_valid(mol)
    print(f"\n分子是否有效: {is_valid}")
    
    validation_info = validator.get_validation_info(mol)
    print("验证信息:")
    for key, value in validation_info.items():
        print(f"  {key}: {value}")
    
    # 测试描述符计算
    descriptors = calculate_descriptors(mol)
    print("\n分子描述符:")
    for key, value in descriptors.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # 测试骨架提取
    scaffold = get_murcko_scaffold(mol)
    print(f"\nMurcko骨架: {scaffold}")
    
    # 测试InChI Key
    inchi_key = get_inchi_key(mol)
    print(f"InChI Key: {inchi_key}")
    
    print("\n化学工具测试完成！")
