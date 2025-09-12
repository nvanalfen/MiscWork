import sys
sys.path.append("..")
import data_utils
from astropy.table import Table
import numpy as np

def test_hdf5_config():
    config_file = "/global/homes/v/vanalfen/MiscWork/skysim_realign/config.yaml"
    config = data_utils.load_yaml_config(config_file)
    test_data = Table(data=[np.array([1,4.5,10,1,4,6.,8],dtype=np.float32),
                            np.array([1,4.5,10,1,4,6.,8],dtype=np.float64),
                            np.array([1,4.5,10,1,4,6.,8],dtype=int),
                           ],
                     names=["A","B","C"])
    print(test_data)
    print(test_data.dtype)
    column_data = {col:test_data[col] for col in test_data.columns}
    
    data_utils.save_hdf5("test.h5", config, column_data)

def build_dummy_config():
    dummy_config = {}

    dummy_config["extra_columns"] = []
    
    dummy_config["alignment_strength"] = {}
    dummy_config["alignment_strength"]["alignment_strength_method"] = "custom"
    dummy_config["alignment_strength"]["custom"] = {}
    dummy_config["alignment_strength"]["sigmoid"] = {}
    dummy_config["alignment_strength"]["custom"]["alignment_strength"] = "some_alignment(colA, colB, colC)"
    dummy_config["alignment_strength"]["sigmoid"] = {"layer1":{}, "layer2":{}, "layer3":{}}
    dummy_config["alignment_strength"]["sigmoid"]["layer1"] = {"x":"lambda colA, colB : colA-colB"}
    dummy_config["alignment_strength"]["sigmoid"]["layer2"] = {"x":"lambda colD:stuff", "x0":"lambda colE:stuff"}
    dummy_config["alignment_strength"]["sigmoid"]["layer3"] = {"x":"lambda colF, colG, colH:stuff"}
    
    dummy_config["alignment_strength_complement"] = {}
    dummy_config["alignment_strength_complement"]["use"] = True
    dummy_config["alignment_strength_complement"]["custom"] = {}
    dummy_config["alignment_strength_complement"]["sigmoid"] = {}
    dummy_config["alignment_strength_complement"]["alignment_strength_method"] = "custom"
    dummy_config["alignment_strength_complement"]["custom"]["alignment_strength"] = "some_alignment(colAA, colBB, colCC)"
    dummy_config["alignment_strength_complement"]["sigmoid"] = {"layer1":{}, "layer2":{}, "layer3":{}}
    dummy_config["alignment_strength_complement"]["sigmoid"]["layer1"] = {"x":"lambda colA, colB : colA-colB"}
    dummy_config["alignment_strength_complement"]["sigmoid"]["layer2"] = {"x":"lambda colDD:stuff"}
    dummy_config["alignment_strength_complement"]["sigmoid"]["layer3"] = {"x":"lambda colFF, colGG, colHH:stuff"}

    return dummy_config

def build_empty_config():
    dummy_config = {}

    dummy_config["extra_columns"] = []
    
    dummy_config["alignment_strength"] = {}
    dummy_config["alignment_strength"]["alignment_strength_method"] = "custom"
    dummy_config["alignment_strength"]["custom"] = {}
    dummy_config["alignment_strength"]["sigmoid"] = {}
    dummy_config["alignment_strength"]["custom"]["alignment_strength"] = ""
    dummy_config["alignment_strength"]["sigmoid"] = {}
 
    dummy_config["alignment_strength_complement"] = {}
    dummy_config["alignment_strength_complement"]["use"] = True
    dummy_config["alignment_strength_complement"]["custom"] = {}
    dummy_config["alignment_strength_complement"]["sigmoid"] = {}
    dummy_config["alignment_strength_complement"]["alignment_strength_method"] = "custom"
    dummy_config["alignment_strength_complement"]["custom"]["alignment_strength"] = ""
    dummy_config["alignment_strength_complement"]["sigmoid"] = {}

    return dummy_config

def test_inherit_alignment_config():
    dummy_config = {"alignment_strength":{}, "alignment_strength_complement":{}}

    dummy_config["alignment_strength"] = {}
    dummy_config["alignment_strength"]["alignment_strength_method"] = "custom"
    dummy_config["alignment_strength"]["custom"] = {}
    dummy_config["alignment_strength"]["sigmoid"] = {}
    dummy_config["alignment_strength"]["sigmoid"]["layer1"] = {"x0":"lambda mag_true_r : 2*mag_true_r","x":0, "k":-0.4,"y_low":-1,"y_high":1, "use_as":None}
    dummy_config["alignment_strength"]["sigmoid"]["layer2"] = {"x0":"lambda mag_true_g : 2*mag_true_g","x":0.2, "k":-0.1,"y_low":-0.8,"y_high":0.8, "use_as":None}
    dummy_config["alignment_strength"]["custom"]["alignment_strength"] = "lambda colA, colB: colA - colB"
    dummy_config["alignment_strength"]["sigmoid"] = {}

    dummy_config["alignment_strength_complement"]["use"] = True

    data_utils.inherit_alignment_config(dummy_config["alignment_strength"], dummy_config["alignment_strength_complement"])

    for key in dummy_config["alignment_strength"]:
        if key == "criteria":
            continue
        assert dummy_config["alignment_strength"][key] == dummy_config["alignment_strength_complement"][key]

    print("PASSED INHERIT ALIGNMENT CONFIG")

def test_extract_columns():
    dummy_config = build_dummy_config()

    # Test using the initial dummy config (no explicit extra columns)
    extra_columns = data_utils.list_extra_columns(dummy_config)
    assert extra_columns == set(["colA", "colB", "colC", "colAA", "colBB", "colCC"])

    # Turn off complement
    dummy_config["alignment_strength_complement"]["use"] = False
    extra_columns = data_utils.list_extra_columns(dummy_config)
    assert extra_columns == set(["colA", "colB", "colC"])

    # Add explicit columns
    dummy_config["extra_columns"] = ["A1", "B2"]
    extra_columns = data_utils.list_extra_columns(dummy_config)
    assert extra_columns == set(["A1", "B2", "colA", "colB", "colC"])

    # Change first method to sigmoid
    dummy_config["alignment_strength"]["alignment_strength_method"] = "sigmoid"
    extra_columns = data_utils.list_extra_columns(dummy_config)
    assert extra_columns == set(["A1", "B2", "colA", "colB", "colD", "colE", "colF", "colG", "colH"])

    # Change second method to sigmoid but keep complement off (no change in result)
    dummy_config["alignment_strength_complement"]["alignment_strength_method"] = "sigmoid"
    extra_columns = data_utils.list_extra_columns(dummy_config)
    assert extra_columns == set(["A1", "B2", "colA", "colB", "colD", "colE", "colF", "colG", "colH"])

    # Change second method to sigmoid (turn complement back on)
    dummy_config["alignment_strength_complement"]["use"] = True
    extra_columns = data_utils.list_extra_columns(dummy_config)
    assert extra_columns == set(["A1", "B2", "colA", "colB", "colD", "colE", "colF", "colG", "colH", "colDD", "colFF", "colGG", "colHH"])

    # Use empty custom
    dummy_config = build_empty_config()
    dummy_config["alignment_strength"]["alignment_strength_method"] = "custom"
    dummy_config["alignment_strength"]["custom"]["alignment_strength"] = ""
    extra_columns = data_utils.list_extra_columns(dummy_config)
    assert len(extra_columns) == 0

    # Use lambda custom with no args
    dummy_config["alignment_strength"]["custom"]["alignment_strength"] = "lambda : None"
    extra_columns = data_utils.list_extra_columns(dummy_config)
    assert len(extra_columns) == 0

    # Use lambda custom with args
    dummy_config["alignment_strength"]["custom"]["alignment_strength"] = "lambda colA, colB : None"
    extra_columns = data_utils.list_extra_columns(dummy_config)
    assert extra_columns == set(["colA", "colB"])

    print("PASSED EXTRACT COLUMNS")

def test_extract_criertia_columns():
    dummy_config = build_empty_config()

    # Add empty criteria
    dummy_config["alignment_strength"]["criteria"] = []
    extra_columns = data_utils.extract_criteria_columns(dummy_config["alignment_strength"]["criteria"])
    assert len(extra_columns) == 0

    # Add null criteria
    dummy_config["alignment_strength"]["criteria"] = [None]
    extra_columns = data_utils.extract_criteria_columns(dummy_config["alignment_strength"]["criteria"])
    assert len(extra_columns) == 0

    # Add non-lambda criteria
    dummy_config["alignment_strength"]["criteria"] = [None, "some_func(A,B)"]
    extra_columns = data_utils.extract_criteria_columns(dummy_config["alignment_strength"]["criteria"])
    assert len(extra_columns) == 0

    # Add real criteria
    dummy_config["alignment_strength"]["criteria"] = [None, "lambda colA, colB: None", "lambda colC : None"]
    extra_columns = data_utils.extract_criteria_columns(dummy_config["alignment_strength"]["criteria"])
    assert extra_columns == set(["colA", "colB", "colC"])

    print("PASSED EXTRACT CRITERIA COLUMNS")

def test_calculate_custom_mu():
    data = Table(data=[np.array([1,1,1,1,1]),
                            np.array([1,2,4,8,16]),
                            np.array([1.1, 1.2, 1.3, 1.4, 1.5]),
                           ],
                     names=["A","B","C"])
    all_mask = np.ones(len(data["A"]), dtype=bool)
    part_mask = np.array([0,1,1,0,1], dtype=bool)
    base_mu = np.zeros(len(all_mask))

    # Test a simple lambda custom
    mus = data_utils.calculate_custom_mu(data, mask=all_mask, custom_script=None, custom_func="lambda A : 2*A")
    copy_mu = np.array(base_mu)
    copy_mu[all_mask] = mus
    assert all(copy_mu == np.array([2,2,2,2,2]))

    # Same test with the partial mask
    mus = data_utils.calculate_custom_mu(data, mask=part_mask, custom_script=None, custom_func="lambda A : 2*A")
    copy_mu = np.array(base_mu)
    copy_mu[part_mask] = mus
    assert all(copy_mu == np.array([0,2,2,0,2]))

    # Use empty function in custom script, full mask
    mus = data_utils.calculate_custom_mu(data, mask=all_mask, custom_script="custom_script", custom_func="test_empty_args()")
    copy_mu = np.array(base_mu)
    copy_mu[all_mask] = mus
    assert all(copy_mu == np.array([1,1,1,1,1]))

    # Use empty function in custom script, partial mask
    mus = data_utils.calculate_custom_mu(data, mask=part_mask, custom_script="custom_script", custom_func="test_empty_args()")
    copy_mu = np.array(base_mu)
    copy_mu[part_mask] = mus
    assert all(copy_mu == np.array([0,1,1,0,1]))

    # Use regular function in custom script, full mask
    mus = data_utils.calculate_custom_mu(data, mask=all_mask, custom_script="custom_script", custom_func="test_mu(B,C)")
    copy_mu = np.array(base_mu)
    copy_mu[all_mask] = mus
    assert all(copy_mu == np.array(data["B"]/data["C"]))

    # Use regular function in custom script, partial mask
    mus = data_utils.calculate_custom_mu(data, mask=part_mask, custom_script="custom_script", custom_func="test_mu(B,C)")
    copy_mu = np.array(base_mu)
    copy_mu[part_mask] = mus
    assert all(copy_mu[part_mask] == np.array(data["B"]/data["C"])[part_mask])
    assert all(copy_mu[~part_mask] == 0)

    print("PASSED CALCULATE CUSTOM MU")

def test_calculate_sigmoid_mu():

    data = Table(data=[np.array([1,1,1,1,1,1,1,1,1,1]),
                            np.array([1,2,4,8,16,16,8,4,2,1]),
                            np.array([1.1, 1.2, 1.3, 1.4, 1.5,1.5,1.4,1.3,1.2,1.1]),
                           ],
                     names=["R","G","Luminosity"])
    layer1 = {"x":"lambda R : 2*R", "x0":-0.5, "k":-1.5, "y_low":-0.2, "y_high":1.}
    layer2 = {"x":"lambda G, R : G-R", "x0":4, "k":-0.4, "y_low":-0.2, "y_high":1.}
    layer3 = {"x":"lambda Luminosity : Luminosity", "x0":-0.5, "k":-1.5, "y_low":-0.2, "y_high":1.}

    base_mus = np.zeros(10)
    maskA = np.array([1,1,1,0,0,1,0,0,0,0], dtype=bool)
    maskB = np.array([0,0,0,1,0,0,0,1,1,0], dtype=bool)

    mus = data_utils.calculate_sigmoid_mu(data, maskA, {"layer1":layer1})
    assert np.allclose(mus, np.array([-0.17242716, -0.17242716, -0.17242716, -0.17242716]))

    mus = data_utils.calculate_sigmoid_mu(data, maskB, {"layer2":layer2})
    assert np.allclose(mus, np.array([0.07777026, 0.51842519, 0.72222974]))

    mus = data_utils.calculate_sigmoid_mu(data, maskA, {"layer2":layer2, "layer3":layer3})
    assert np.allclose(mus, np.array([-0.11695854, -0.13320614, -0.15475835, -0.19930976]))

    print("PASSED CALCULATE SIGMOID MU")

def test_calculate_mus():
    data = Table(data=[np.array([1,1,1,1,1,1,1,1,1,1]),
                            np.array([1,2,4,8,16,16,8,4,2,1]),
                            np.array([1.1, 1.2, 1.3, 1.4, 1.5,1.5,1.4,1.3,1.2,1.1]),
                           ],
                     names=["A","B","C"])
    base_mus = np.zeros(10)
    maskA = np.array([1,1,1,0,0,1,0,0,0,0], dtype=bool)
    maskB = np.array([0,0,0,1,0,0,0,1,1,0], dtype=bool)
    total_mask = maskA|maskB

    dummy_config = build_empty_config()

    # Make both masks use the same custom function for both
    dummy_config["alignment_strength"]["custom"]["custom_script"] = "custom_script"
    dummy_config["alignment_strength"]["custom"]["alignment_strength"] = "test_mu(B,C)"
    dummy_config["alignment_strength_complement"]["use"] = True
    del dummy_config["alignment_strength_complement"]["custom"]
    mus = data_utils.calculate_mus(dummy_config, data, maskA, maskB)
    assert all(mus[total_mask] == (data["B"]/data["C"])[total_mask])
    assert all(mus[~total_mask] == base_mus[~total_mask])

    # Different custom functions for each mask (deliberately leave script name out, should inherit)
    dummy_config["alignment_strength_complement"]["custom"] = {}
    dummy_config["alignment_strength_complement"]["custom"]["alignment_strength"] = "test_empty_args()"
    mus = data_utils.calculate_mus(dummy_config, data, maskA, maskB)
    assert all(mus[maskA] == (data["B"]/data["C"])[maskA])
    assert all(mus[maskB] == 1)
    assert all(mus[~total_mask] == base_mus[~total_mask])

    # Cconstant method for complement
    dummy_config["alignment_strength_complement"]["alignment_strength_method"] = "constant"
    dummy_config["alignment_strength_complement"]["constant"] = 2.
    mus = data_utils.calculate_mus(dummy_config, data, maskA, maskB)
    assert all(mus[maskA] == (data["B"]/data["C"])[maskA])
    assert all(mus[maskB] == 2)
    assert all(mus[~total_mask] == base_mus[~total_mask])

    # Turn off complement
    dummy_config["alignment_strength_complement"]["use"] = False
    mus = data_utils.calculate_mus(dummy_config, data, maskA, None)
    assert all(mus[maskA] == (data["B"]/data["C"])[maskA])
    assert all(mus[~maskA] == base_mus[~maskA])

    print("TODO: SIGMOIDS")
    print("PASSED CALCULATE MUS")

def test_remap_mu_tidal():
    mus = np.array([1,2,3,4,5,6,7,8,9,10]).astype(float)

    maskA = np.array([1,1,1,0,0,0,1,0,1,0], dtype=bool)
    maskB = np.array([0,0,0,1,1,0,0,1,0,0], dtype=bool)

    # No remapping
    new_mus = np.array(mus)
    data_utils.remap_mu_tidal(new_mus,maskA,False)
    assert all(new_mus == mus)

    # Simple tidal remapping
    new_mus = np.array(mus)
    data_utils.remap_mu_tidal(new_mus,maskA,True)
    assert all(new_mus[maskA] == mus[maskA]*0.67)
    assert all(new_mus[~maskA] == mus[~maskA])

    print("PASSED REMAP MU")

if __name__ == "__main__":
    test_extract_columns()
    test_extract_criertia_columns()
    test_calculate_custom_mu()
    test_calculate_mus()
    test_inherit_alignment_config()
    test_calculate_sigmoid_mu()
    test_remap_mu_tidal()

    print("\n>>>>>PASSED ALL TESTS!")