# ALF World Setup

## Environment Setup

ALF World requires a separate environment.

```
conda env create -f alf_conda.yml
conda activate vrenv-alf
pip install -e ../LLaVA
pip install -e ../gym-cards
pip install git+https://github.com/openai/CLIP.git
pip3 install numpy==1.23.5
pip3 install protobuf==3.20.3
pip3 install pydantic==1.10.14
pip3 install pydantic-core==2.16.3
pip3 uninstall frozenlist gradio murmurhash preshed spacy srsly thinc weasel aiosignal annotated-types blis catalogue cloudpathlib cymem
export ALFWORLD_DATA=<storage_path>
alfworld-download
```

Test the installation of alfworld through the following two commands

```
alfworld-play-tw
alfworld-play-thor
```

Make sure the version of alfworld `>= 0.3.2`

## Reproduction

```bash
conda activate vrenv-alf
cd scripts
bash run_alf.sh
```
