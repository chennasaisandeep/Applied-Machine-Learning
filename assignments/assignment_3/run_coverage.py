# run_coverage.py
import os

def run_coverage():
    """Run tests with coverage and save results to coverage.txt"""
    # Use os.system to run pytest with coverage report generation
    command = "pytest -v --cov=score --cov=app --cov-report=term test.py > coverage.txt"
    exit_code = os.system(command)
    
    if exit_code == 0:
        print("Tests completed successfully. Coverage report saved to coverage.txt")
    else:
        print(f"Tests failed with exit code {exit_code//256}. Check coverage.txt for details.")

if __name__ == "__main__":
    run_coverage()