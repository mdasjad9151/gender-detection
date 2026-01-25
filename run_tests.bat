@echo off
echo ==================================
echo Gender Detection System - Testing
echo ==================================
echo.

echo Installing test dependencies...
pip install -r requirements-test.txt

echo.
echo ==================================
echo Running Unit Tests
echo ==================================
pytest tests/ -v -m "not integration"

echo.
echo ==================================
echo Running Integration Tests
echo ==================================
pytest tests/ -v -m integration

echo.
echo ==================================
echo Running All Tests with Coverage
echo ==================================
pytest tests/ -v --cov=src --cov-report=term --cov-report=html

echo.
echo ==================================
echo Coverage Report Generated
echo ==================================
echo HTML report available at: htmlcov\index.html
echo.

pause