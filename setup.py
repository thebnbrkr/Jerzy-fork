from setuptools import setup, find_packages

setup(
    name="jerzy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["openai", "tenacity"],
    author="Anirudh Anil",
    description="Jerzy: A modular, explainable agent framework for LLMs.",
    long_description="Minimal LLM framework for reasoning, memory, and tools.",
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)
