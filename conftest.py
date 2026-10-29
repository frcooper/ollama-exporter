"""Pytest configuration.

Its only job is to sit at the repository root: pytest prepends the directory
holding this file to ``sys.path``, which is how ``tests/`` imports
``ollama_exporter`` without the project being installed as a package.
"""
