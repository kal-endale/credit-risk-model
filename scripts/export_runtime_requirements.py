"""Export exact API runtime dependency versions."""

from importlib.metadata import version
from pathlib import Path


RUNTIME_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "threadpoolctl",
    "fastapi",
    "uvicorn",
    "pydantic",
]


def main() -> None:
    """Write reproducible API requirements."""

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "requirements-api.txt"

    requirements = [
        f"{package}=={version(package)}"
        for package in RUNTIME_PACKAGES
    ]

    output_path.write_text(
        "\n".join(requirements) + "\n",
        encoding="utf-8",
    )

    print("API runtime requirements")
    print("-" * 50)
    print("\n".join(requirements))
    print(f"\nOutput: {output_path.name}")


if __name__ == "__main__":
    main()