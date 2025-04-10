# config.py - Configuration management for mandible FEA pipeline

import os
import yaml
import logging

logger = logging.getLogger("mandible_fea.config")


def load_config(config_path):
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def validate_config(config):
    """Validate configuration structure and required fields."""
    logger.info("Validating configuration")

    # Check for required sections
    required_sections = ["global", "cases"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in configuration: {section}")

    # Check for required fields in each case
    for i, case in enumerate(config["cases"]):
        required_case_fields = ["dicom_dir", "mask_path"]
        for field in required_case_fields:
            if field not in case:
                raise ValueError(f"Missing required field '{field}' in case {i}")

    # Validate paths exist
    for case in config["cases"]:
        if not os.path.exists(case["dicom_dir"]):
            logger.warning(f"DICOM directory does not exist: {case['dicom_dir']}")
        if not os.path.exists(case["mask_path"]):
            logger.warning(f"Mask file does not exist: {case['mask_path']}")

    return True


def generate_parameter_sweep(config, sweep_config):
    """Generate configurations for parameter sweep."""
    logger.info("Generating parameter sweep configurations")

    # Find base case
    base_case_name = sweep_config["base_case"]
    base_case = None
    for case in config["cases"]:
        if case.get("name") == base_case_name:
            base_case = case.copy()
            break

    if base_case is None:
        raise ValueError(f"Base case not found for parameter sweep: {base_case_name}")

    # Generate parameter combinations
    import itertools

    param_names = []
    param_values = []

    for param_name, values in sweep_config["parameters"].items():
        param_names.append(param_name)
        param_values.append(values)

    sweep_cases = []
    for i, combination in enumerate(itertools.product(*param_values)):
        case = base_case.copy()
        case["name"] = f"{base_case_name}_sweep_{i + 1}"

        # Apply parameter combination
        for j, param_name in enumerate(param_names):
            param_value = combination[j]

            # Handle nested parameters
            if "." in param_name:
                parts = param_name.split(".")
                target = case
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = param_value
            else:
                case[param_name] = param_value

        sweep_cases.append(case)

    return sweep_cases


def create_example_config(output_path):
    """Create an example configuration file."""
    logger.info(f"Creating example configuration file at {output_path}")

    example_config = {
        "global": {"output_base_dir": "./output", "num_workers": 8},
        "materials": {
            "calibration": {
                "hu_points": [-1000, 1200, 1800],
                "rho_points": [0.001, 2.2, 2.9],
            },
            "models": {
                "helgason": {"a": 6850, "b": 1.49},
                "carter_hayes": {"a": 3790, "b": 1.56},
            },
        },
        "export": {"vtk": True, "inp": True, "dat": True, "modulus_bins": 10},
        "cases": [
            {
                "name": "patient_001",
                "dicom_dir": "/path/to/dicom/patient_001",
                "mask_path": "/path/to/masks/patient_001.nii",
                "material_model": "helgason",
            },
            {
                "name": "patient_002",
                "dicom_dir": "/path/to/dicom/patient_002",
                "mask_path": "/path/to/masks/patient_002.nii",
                "material_model": "carter_hayes",
            },
        ],
        "parameter_sweeps": [
            {
                "base_case": "patient_001",
                "parameters": {
                    "material_model": ["helgason", "carter_hayes"],
                    "modulus_bins": [5, 10, 20],
                },
            }
        ],
    }

    try:
        with open(output_path, "w") as f:
            yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        logger.error(f"Error creating example configuration: {str(e)}")
        raise
