import mordred
import json
from mordred import ABCIndex, AcidBase, AdjacencyMatrix, Aromatic, AtomCount, Autocorrelation, BCUT, BalabanJ, BaryszMatrix, \
    BertzCT, BondCount, CPSA, CarbonTypes, Chi, Constitutional, DetourMatrix, DistanceMatrix, EccentricConnectivityIndex, \
        ExtendedTopochemicalAtom, FragmentComplexity, Framework, GeometricalIndex, GravitationalIndex, HydrogenBond, \
        InformationContent, KappaShapeIndex, Lipinski, LogS, McGowanVolume, MoRSE, MoeType, MolecularDistanceEdge, MolecularId, \
        MomentOfInertia, PBF, PathCount, Polarizability, RingCount, RotatableBond, SLogP, TopoPSA, TopologicalCharge, TopologicalIndex, \
        VdwVolumeABC, VertexAdjacencyInformation, WalkCount, Weight, WienerIndex, ZagrebIndex
        
descriptor_classes_str = ["ABCIndex", "AcidBase", "AdjacencyMatrix", "Aromatic", "AtomCount", "Autocorrelation", "BCUT", "BalabanJ", 
                      "BaryszMatrix", "BertzCT", "BondCount", "CPSA", "CarbonTypes", "Chi", "Constitutional", "DetourMatrix", 
                      "DistanceMatrix", "EccentricConnectivityIndex", "ExtendedTopochemicalAtom", "FragmentComplexity", 
                      "Framework", "GeometricalIndex", "GravitationalIndex", "HydrogenBond", "InformationContent", "KappaShapeIndex", 
                      "Lipinski", "LogS", "McGowanVolume", "MoRSE", "MoeType", "MolecularDistanceEdge", "MolecularId", "MomentOfInertia", "PBF", 
                      "PathCount", "Polarizability", "RingCount", "RotatableBond", "SLogP", "TopoPSA", "TopologicalCharge", "TopologicalIndex", 
                      "VdwVolumeABC", "VertexAdjacencyInformation", "WalkCount", "Weight", "WienerIndex", "ZagrebIndex"]

descriptor_classes = [ABCIndex, AcidBase, AdjacencyMatrix, Aromatic, AtomCount, Autocorrelation, BCUT, BalabanJ, 
                      BaryszMatrix, BertzCT, BondCount, CPSA, CarbonTypes, Chi, Constitutional, DetourMatrix, 
                      DistanceMatrix, EccentricConnectivityIndex, ExtendedTopochemicalAtom, FragmentComplexity, 
                      Framework, GeometricalIndex, GravitationalIndex, HydrogenBond, InformationContent, KappaShapeIndex, Lipinski, 
                      LogS, McGowanVolume, MoRSE, MoeType, MolecularDistanceEdge, MolecularId, MomentOfInertia, PBF, PathCount, 
                      Polarizability, RingCount, RotatableBond, SLogP, TopoPSA, TopologicalCharge, TopologicalIndex, 
                      VdwVolumeABC, VertexAdjacencyInformation, WalkCount, Weight, WienerIndex, ZagrebIndex]

from pathlib import Path

CURR_DIR = Path(__file__).parent
mordred_version = mordred.__version__
MOLECULAR_DESCRIPTORS_FEATURE_NAMES_FILE = "molecular_descriptors_feature_names"+"_"+mordred_version+".json"
MOLECULAR_META_DESCRIPTORS_FEATURE_NAMES_FILE = "molecular_meta_descriptors_feature_names"+"_"+mordred_version+".json"


def get_mordred_descriptors_definition():
    descriptors_definition = {}
    for item1, item2 in zip(descriptor_classes_str, descriptor_classes):
        descriptors_definition[item1] = [item2]
        
    molecular_descriptors_custom = {
        "QED": [SLogP, RingCount.RingCount(aromatic=True), HydrogenBond.HBondAcceptor, HydrogenBond.HBondDonor, RotatableBond.RotatableBondsCount, Weight.Weight(exact=True), TopoPSA.TopoPSA(no_only=False)]
    }
    
    descriptors_definition.update(molecular_descriptors_custom)
    molecular_meta_descriptors_definition = {
        "QED": ["QED"]
    }
    
    return descriptors_definition, molecular_meta_descriptors_definition

def get_mordred_feature_names():
    with open((CURR_DIR/MOLECULAR_DESCRIPTORS_FEATURE_NAMES_FILE).as_posix(), "r") as f:
        molecular_descriptors_feature_names = json.load(f)
    with open((CURR_DIR/MOLECULAR_META_DESCRIPTORS_FEATURE_NAMES_FILE).as_posix(), "r") as f:
        molecular_meta_descriptors_feature_names = json.load(f)
        
    return molecular_descriptors_feature_names, molecular_meta_descriptors_feature_names
    