#!/usr/bin/env python
"""
Cross-platform test runner script
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print results"""
    print("\n" + "=" * 70)
    print(f" {description}")
    print("=" * 70)
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n {description} failed!")
        return False
    
    print(f"\n✓ {description} completed successfully!")
    return True


def main():
    """Main test runner"""
    print("=" * 70)
    print(" Gender Detection System - Test Suite")
    print("=" * 70)
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("\n Error: 'tests' directory not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Install dependencies
    print("\n Installing test dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"])
    
    # Run different test suites
    all_passed = True
    
    # 1. Unit tests only
    all_passed &= run_command(
        f'{sys.executable} -m pytest tests/ -v -m "not integration"',
        "Unit Tests"
    )
    
    # 2. Integration tests
    all_passed &= run_command(
        f'{sys.executable} -m pytest tests/ -v -m integration',
        "Integration Tests"
    )
    
    # 3. All tests with coverage
    all_passed &= run_command(
        f'{sys.executable} -m pytest tests/ -v --cov=src --cov-report=term --cov-report=html --cov-report=xml',
        "Complete Test Suite with Coverage"
    )
    
    # Summary
    print("\n" + "=" * 70)
    print(" Test Summary")
    print("=" * 70)
    
    if all_passed:
        print("\n All tests passed!")
    else:
        print("\n Some tests failed. Please check the output above.")
    
    # Coverage report location
    if Path("htmlcov/index.html").exists():
        print("\n Coverage Report:")
        print(f"   HTML: {Path('htmlcov/index.html').absolute()}")
    
    if Path("coverage.xml").exists():
        print(f"   XML:  {Path('coverage.xml').absolute()}")
    
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())