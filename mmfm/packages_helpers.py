import subprocess
import sys

from mmfm.config import cnf


def install(package, version=None):
    if version:
        package_with_version = f"{package}=={version}"
    else:
        package_with_version = package
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package_with_version]
    )

def download_spacy_model(model_name):
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
    except subprocess.CalledProcessError as e:
        print(f"Failed to download spaCy model {model_name}: {e}")
        raise



def install_all_packages():
    # subprocess.check_call(
    #     [
    #         sys.executable,
    #         "-m",
    #         "pip",
    #         "install",
    #         "torch",
    #         "torchvision",
    #         "--index-url",
    #         "https://download.pytorch.org/whl/cpu",
    #     ]
    # )
    if cnf.packages:
        for package_info in cnf.packages:
            package_name = package_info["name"]
            version = package_info.get("version")  # Check if version is specified
            try:
                install(package_name, version)
            except subprocess.CalledProcessError as e:
                print(f"Failed to install package {package_name}: {e}")
                raise

        # Install the spaCy model after installing spaCy
        # download_spacy_model("en_core_web_sm")
