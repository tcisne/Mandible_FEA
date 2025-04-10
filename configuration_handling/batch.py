# batch.py - Batch processing for mandible FEA pipeline

import os
import logging
import time
from concurrent.futures import ProcessPoolExecutor
import sys

# Add parent directory to path to import core and config modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import process_single_case
from processing_functions.config import (
    load_config,
    validate_config,
    generate_parameter_sweep,
)

logger = logging.getLogger("mandible_fea.batch")


def process_batch(config_path):
    """Process a batch of cases defined in a configuration file."""
    start_time = time.time()
    logger.info(f"Starting batch processing with config: {config_path}")

    # Load and validate configuration
    config = load_config(config_path)
    validate_config(config)

    # Prepare cases list
    all_cases = config["cases"].copy()

    # Add parameter sweep cases if defined
    if "parameter_sweeps" in config:
        for sweep_config in config["parameter_sweeps"]:
            sweep_cases = generate_parameter_sweep(config, sweep_config)
            all_cases.extend(sweep_cases)

    # Get number of workers
    num_workers = config["global"].get("num_workers", os.cpu_count())
    logger.info(f"Using {num_workers} workers for parallel processing")

    # Process cases in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_case = {
            executor.submit(process_single_case, case, config["global"]): case
            for case in all_cases
        }

        for future in future_to_case:
            case = future_to_case[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed case: {case.get('name', 'unnamed')}")
            except Exception as e:
                logger.error(
                    f"Error processing case {case.get('name', 'unnamed')}: {str(e)}"
                )
                results.append(
                    {
                        "case": case.get("name", "unnamed"),
                        "status": "error",
                        "error": str(e),
                    }
                )

    # Generate summary
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")

    elapsed_time = time.time() - start_time
    logger.info(f"Batch processing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Summary: {success_count} succeeded, {error_count} failed")

    return results


def generate_report(results, output_path):
    """Generate a summary report of batch processing results."""
    logger.info(f"Generating report at {output_path}")

    with open(output_path, "w") as f:
        f.write("# Mandible FEA Batch Processing Report\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        f.write(f"- Total cases: {len(results)}\n")
        f.write(f"- Successful: {success_count}\n")
        f.write(f"- Failed: {error_count}\n\n")

        f.write("## Case Details\n\n")
        for result in results:
            f.write(f"### Case: {result['case']}\n")
            f.write(f"- Status: {result['status']}\n")
            if result["status"] == "success":
                f.write(f"- Output directory: {result['output_dir']}\n")
            else:
                f.write(f"- Error: {result['error']}\n")
            f.write("\n")


def process_sequential_batch(config_path):
    """Process a batch of cases sequentially (no parallelization)."""
    start_time = time.time()
    logger.info(f"Starting sequential batch processing with config: {config_path}")

    # Load and validate configuration
    config = load_config(config_path)
    validate_config(config)

    # Prepare cases list
    all_cases = config["cases"].copy()

    # Add parameter sweep cases if defined
    if "parameter_sweeps" in config:
        for sweep_config in config["parameter_sweeps"]:
            sweep_cases = generate_parameter_sweep(config, sweep_config)
            all_cases.extend(sweep_cases)

    # Process cases sequentially
    results = []
    for case in all_cases:
        try:
            logger.info(f"Processing case: {case.get('name', 'unnamed')}")
            result = process_single_case(case, config["global"])
            results.append(result)
            logger.info(f"Completed case: {case.get('name', 'unnamed')}")
        except Exception as e:
            logger.error(
                f"Error processing case {case.get('name', 'unnamed')}: {str(e)}"
            )
            results.append(
                {
                    "case": case.get("name", "unnamed"),
                    "status": "error",
                    "error": str(e),
                }
            )

    # Generate summary
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")

    elapsed_time = time.time() - start_time
    logger.info(f"Sequential batch processing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Summary: {success_count} succeeded, {error_count} failed")

    return results
