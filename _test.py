import pytest
from src.pipline.training_pipeline import TrainPipeline

# Pytest only collects functions or classes that follow its naming convention:

# Functions must start with test_

# Classes must start with Test and methods inside must start with test_


def test_pipeline_runs():
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
    # Add an assertion so pytest knows it's a test
    assert True