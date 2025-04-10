# core.py - Core functionality for mandible FEA pipeline

import os
import numpy as np
import SimpleITK as sitk
import pyvista as pv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mandible_fea")


def hu_to_density(hu, calib_coeffs=None):
    """Convert HU values to density using calibration coefficients."""
    if calib_coeffs is None:
        # Default calibration
        hu_points = np.array([-1000, 1200, 1800])
        rho_points = np.array([0.001, 2.2, 2.9])
        calib_coeffs = np.polyfit(hu_points, rho_points, deg=1)

    return calib_coeffs[0] * hu + calib_coeffs[1]


def density_to_modulus(density, model="helgason", params=None):
    """Convert density to Young's modulus using specified model."""
    if params is None:
        params = {}

    if model == "helgason":
        a = params.get("a", 6850)
        b = params.get("b", 1.49)
        return a * density**b
    elif model == "carter_hayes":
        a = params.get("a", 3790)
        b = params.get("b", 1.56)
        return a * density**b
    else:
        raise ValueError(f"Unsupported model: {model}")


def load_dicom_volume(dicom_dir):
    """Load DICOM volume from directory."""
    logger.info(f"Loading DICOM volume from {dicom_dir}")
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        if not dicom_names:
            raise ValueError(f"No DICOM files found in {dicom_dir}")

        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return sitk.GetArrayFromImage(image), image
    except Exception as e:
        logger.error(f"Error loading DICOM volume: {str(e)}")
        raise


def load_mask(mask_path):
    """Load segmentation mask."""
    logger.info(f"Loading mask from {mask_path}")
    try:
        mask_img = sitk.ReadImage(mask_path)
        return sitk.GetArrayFromImage(mask_img)
    except Exception as e:
        logger.error(f"Error loading mask: {str(e)}")
        raise


def process_single_case(case_config, global_config):
    """Process a single case with the given configuration."""
    try:
        logger.info(f"Processing case: {case_config.get('name', 'unnamed')}")

        # Extract parameters
        dicom_dir = case_config["dicom_dir"]
        mask_path = case_config["mask_path"]
        output_dir = os.path.join(
            global_config.get("output_base_dir", "."),
            case_config.get("name", "unnamed"),
        )
        os.makedirs(output_dir, exist_ok=True)

        # Material model parameters
        material_model = case_config.get("material_model", "helgason")
        material_params = (
            global_config.get("materials", {}).get("models", {}).get(material_model, {})
        )

        # Calibration parameters
        calib_params = global_config.get("materials", {}).get("calibration", {})
        hu_points = np.array(calib_params.get("hu_points", [-1000, 1200, 1800]))
        rho_points = np.array(calib_params.get("rho_points", [0.001, 2.2, 2.9]))
        calib_coeffs = np.polyfit(hu_points, rho_points, deg=1)

        # Export parameters
        modulus_bins = global_config.get("export", {}).get("modulus_bins", 10)

        # Load data
        ct_array, ct_image = load_dicom_volume(dicom_dir)
        mask_array = load_mask(mask_path)

        spacing = ct_image.GetSpacing()[::-1]
        origin = ct_image.GetOrigin()[::-1]

        # Process data
        logger.info("Calibrating HU to modulus...")
        hu_array = ct_array.astype(np.float32)
        rho_array = hu_to_density(hu_array, calib_coeffs)
        E_array = density_to_modulus(
            rho_array, model=material_model, params=material_params
        )
        E_masked = np.where(mask_array > 0, E_array, 0)

        # Export results
        if global_config.get("export", {}).get("vtk", True):
            export_to_vtk(
                E_masked, spacing, origin, os.path.join(output_dir, "modulus_field.vtk")
            )

        if global_config.get("export", {}).get("inp", True) or global_config.get(
            "export", {}
        ).get("dat", True):
            inp_path = (
                os.path.join(output_dir, "mandible_model.inp")
                if global_config.get("export", {}).get("inp", True)
                else None
            )
            dat_path = (
                os.path.join(output_dir, "mandible_model.dat")
                if global_config.get("export", {}).get("dat", True)
                else None
            )
            export_to_fea(E_masked, spacing, origin, inp_path, dat_path, modulus_bins)

        logger.info(f"Case {case_config.get('name', 'unnamed')} processed successfully")
        return {
            "case": case_config.get("name", "unnamed"),
            "status": "success",
            "output_dir": output_dir,
        }
    except Exception as e:
        logger.error(
            f"Error processing case {case_config.get('name', 'unnamed')}: {str(e)}"
        )
        return {
            "case": case_config.get("name", "unnamed"),
            "status": "error",
            "error": str(e),
        }


def export_to_vtk(E_array, spacing, origin, vtk_path):
    """Export modulus field to VTK format."""
    logger.info(f"Creating VTK volume at {vtk_path}")
    try:
        grid = pv.UniformGrid()
        grid.dimensions = np.array(E_array.shape)[::-1] + 1
        grid.origin = origin
        grid.spacing = spacing
        grid.cell_data["Youngs_Modulus"] = E_array.flatten(order="F")
        grid.save(vtk_path)
    except Exception as e:
        logger.error(f"Error exporting to VTK: {str(e)}")
        raise


def export_to_fea(
    E_array, spacing, origin, inp_path=None, dat_path=None, modulus_bins=10
):
    """Export to FEA formats (Abaqus INP and/or MSC Marc DAT)."""
    logger.info("Exporting to FEA formats")
    try:
        nz, ny, nx = E_array.shape
        dx, dy, dz = spacing

        node_id = 1
        elem_id = 1
        node_map = {}
        nodes = []
        elements = []
        elem_sets = {i: [] for i in range(modulus_bins)}

        E_flat = E_array.flatten()
        nonzero_indices = np.argwhere(E_array > 0)

        E_min, E_max = E_flat[E_flat > 0].min(), E_flat.max()
        bin_edges = np.linspace(E_min, E_max, modulus_bins + 1)

        # Process nodes and elements
        for idx in nonzero_indices:
            z, y, x = idx
            for dz_ in [0, 1]:
                for dy_ in [0, 1]:
                    for dx_ in [0, 1]:
                        p = (z + dz_, y + dy_, x + dx_)
                        if p not in node_map:
                            X = origin[2] + (x + dx_) * dx
                            Y = origin[1] + (y + dy_) * dy
                            Z = origin[0] + (z + dz_) * dz
                            node_map[p] = node_id
                            nodes.append((node_id, X, Y, Z))
                            node_id += 1

        # Process elements and assign to material sets
        for idx in nonzero_indices:
            z, y, x = idx
            n1 = node_map[(z, y, x)]
            n2 = node_map[(z, y, x + 1)]
            n3 = node_map[(z, y + 1, x + 1)]
            n4 = node_map[(z, y + 1, x)]
            n5 = node_map[(z + 1, y, x)]
            n6 = node_map[(z + 1, y, x + 1)]
            n7 = node_map[(z + 1, y + 1, x + 1)]
            n8 = node_map[(z + 1, y + 1, x)]
            elements.append((elem_id, n1, n2, n3, n4, n5, n6, n7, n8))

            E_val = E_array[z, y, x]
            bin_idx = get_material_bin(E_val, bin_edges)
            elem_sets[bin_idx].append(elem_id)
            elem_id += 1

        # Write INP file if requested
        if inp_path:
            write_inp_file(inp_path, nodes, elements, elem_sets, bin_edges)

        # Write DAT file if requested
        if dat_path:
            write_dat_file(dat_path, bin_edges)

    except Exception as e:
        logger.error(f"Error exporting to FEA formats: {str(e)}")
        raise


def get_material_bin(E, bin_edges):
    """Determine which material bin a modulus value belongs to."""
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= E < bin_edges[i + 1]:
            return i
    return len(bin_edges) - 2


def write_inp_file(inp_path, nodes, elements, elem_sets, bin_edges):
    """Write Abaqus INP file."""
    logger.info(f"Writing INP file to {inp_path}")
    with open(inp_path, "w") as f:
        f.write("*Heading\n** Generated by mandible_fea_pipeline.py\n*Node\n")
        for n in nodes:
            f.write(f"{n[0]}, {n[1]:.3f}, {n[2]:.3f}, {n[3]:.3f}\n")

        f.write("*Element, type=C3D8\n")
        for e in elements:
            f.write(
                f"{e[0]}, {e[1]}, {e[2]}, {e[3]}, {e[4]}, {e[5]}, {e[6]}, {e[7]}, {e[8]}\n"
            )

        for i, elems in elem_sets.items():
            if elems:
                f.write(f"*Elset, elset=Material_{i + 1}\n")
                for j in range(0, len(elems), 8):
                    line = ", ".join(map(str, elems[j : j + 8]))
                    f.write(line + "\n")

        for i in range(len(bin_edges) - 1):
            E_bin = 0.5 * (bin_edges[i] + bin_edges[i + 1])
            f.write(f"*Material, name=Material_{i + 1}\n")
            f.write(f"*Elastic\n{E_bin:.2f}, 0.30\n")


def write_dat_file(dat_path, bin_edges):
    """Write MSC Marc DAT file."""
    logger.info(f"Writing DAT file to {dat_path}")
    with open(dat_path, "w") as f:
        f.write("$ Generated by mandible_fea_pipeline.py\n$")
        f.write("$ Node and element data can be converted from INP\n")
        f.write("$ Material definitions:\n")
        for i in range(len(bin_edges) - 1):
            E_bin = 0.5 * (bin_edges[i] + bin_edges[i + 1])
            f.write(f"MATERIAL, NAME=Material_{i + 1}\n")
            f.write(f"ELASTIC, {E_bin:.2f}, 0.30\n")
