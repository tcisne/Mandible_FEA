# Mandible FEA Pipeline

A pipeline for processing DICOM images and segmentation masks to create finite element analysis (FEA) models for mandible simulations.

## Features

- Process single cases or batches of multiple cases
- Support for parameter sweeps for sensitivity analysis
- Parallel processing for improved performance
- Configuration via YAML files
- Export to VTK, Abaqus INP, and MSC Marc DAT formats

## Requirements

- Python 3.7+
- NumPy
- SimpleITK
- PyVista
- PyYAML
- SciPy

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mandible_fea.git
   cd mandible_fea
   ```

2. Install dependencies:
   ```
   pip install numpy simpleitk pyvista pyyaml scipy
   ```

## Usage

### Single Case Processing

To process a single case:

```bash
python processing_functionality/mandible_fea_pipeline.py --single --dicom_dir /path/to/dicom --mask /path/to/mask.nii --out /path/to/output
```

Optional arguments:
- `--material_model`: Material model to use (default: "helgason")
- `--log`: Logging level (default: "INFO")

### Batch Processing

To process multiple cases defined in a configuration file:

```bash
python processing_functionality/mandible_fea_pipeline.py --batch --config /path/to/config.yaml
```

Optional arguments:
- `--report`: Path to save a report of the batch processing results
- `--sequential`: Process cases sequentially (no parallelization)
- `--log`: Logging level (default: "INFO")

### Creating an Example Configuration

To create an example configuration file:

```bash
python processing_functionality/mandible_fea_pipeline.py --create-config --config /path/to/output.yaml
```

## Configuration File Format

The configuration file uses YAML format and has the following structure:

```yaml
# Global configuration
global:
  output_base_dir: "./output"
  num_workers: 8  # Number of parallel workers
  
# Material models and calibration
materials:
  calibration:
    hu_points: [-1000, 1200, 1800]
    rho_points: [0.001, 2.2, 2.9]
  models:
    helgason:
      a: 6850
      b: 1.49
    carter_hayes:
      a: 3790
      b: 1.56

# Export settings
export:
  vtk: true
  inp: true
  dat: true
  modulus_bins: 10

# Cases to process
cases:
  - name: "patient_001"
    dicom_dir: "/path/to/dicom/patient_001"
    mask_path: "/path/to/masks/patient_001.nii"
    material_model: "helgason"
  - name: "patient_002"
    dicom_dir: "/path/to/dicom/patient_002"
    mask_path: "/path/to/masks/patient_002.nii"
    material_model: "carter_hayes"

# Parameter sweep configurations
parameter_sweeps:
  - base_case: "patient_001"
    parameters:
      material_model: ["helgason", "carter_hayes"]
      modulus_bins: [5, 10, 20]
```

## Project Structure

- `core.py`: Core processing functions
- `processing_functions/config.py`: Configuration handling
- `configuration_handling/batch.py`: Batch processing functionality
- `processing_functionality/mandible_fea_pipeline.py`: Main entry point

## License

[MIT License](LICENSE)