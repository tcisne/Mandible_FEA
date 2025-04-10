#!/usr/bin/env python
# mandible_fea_pipeline.py - Main entry point for mandible FEA pipeline

import os
import sys
import argparse
import logging

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import process_single_case
from processing_functions.config import (
    load_config,
    validate_config,
    create_example_config,
)
from configuration_handling.batch import (
    process_batch,
    generate_report,
    process_sequential_batch,
)


def main():
    """Main entry point for the mandible FEA pipeline."""
    parser = argparse.ArgumentParser(description="Mandible FEA pipeline")

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--single", action="store_true", help="Process a single case"
    )
    mode_group.add_argument(
        "--batch", action="store_true", help="Process a batch of cases"
    )
    mode_group.add_argument(
        "--create-config",
        action="store_true",
        help="Create an example configuration file",
    )

    # Single mode arguments
    parser.add_argument("--dicom_dir", help="Path to DICOM folder (single mode)")
    parser.add_argument("--mask", help="Path to segmentation mask (single mode)")
    parser.add_argument("--out", help="Output directory (single mode)")
    parser.add_argument(
        "--material_model", default="helgason", help="Material model (single mode)"
    )

    # Batch mode arguments
    parser.add_argument("--config", help="Path to configuration file (batch mode)")
    parser.add_argument("--report", help="Path to save report (batch mode)")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process batch sequentially (no parallelization)",
    )

    # Common arguments
    parser.add_argument("--log", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("mandible_fea")

    try:
        if args.create_config:
            # Create example configuration file
            output_path = args.config or "example_config.yaml"
            create_example_config(output_path)
            logger.info(f"Example configuration file created at {output_path}")
            return 0

        elif args.single:
            # Validate single mode arguments
            if not args.dicom_dir or not args.mask or not args.out:
                parser.error("Single mode requires --dicom_dir, --mask, and --out")

            # Process single case
            os.makedirs(args.out, exist_ok=True)

            case_config = {
                "name": "single_case",
                "dicom_dir": args.dicom_dir,
                "mask_path": args.mask,
                "material_model": args.material_model,
            }

            global_config = {"output_base_dir": args.out}

            result = process_single_case(case_config, global_config)

            if result["status"] == "success":
                logger.info("Single case processed successfully")
                return 0
            else:
                logger.error(f"Error processing single case: {result['error']}")
                return 1

        elif args.batch:
            # Validate batch mode arguments
            if not args.config:
                parser.error("Batch mode requires --config")

            # Process batch
            if args.sequential:
                results = process_sequential_batch(args.config)
            else:
                results = process_batch(args.config)

            # Generate report if requested
            if args.report:
                generate_report(results, args.report)

            # Check if all cases were successful
            if all(r["status"] == "success" for r in results):
                logger.info("All cases processed successfully")
                return 0
            else:
                logger.warning("Some cases failed during processing")
                return 1

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
