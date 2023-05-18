# PersonaChatGen

**This is the official github repository for [PERSONACHATGEN: Generating Personalized Dialogues using GPT-3](https://aclanthology.org/2022.ccgpk-1.4/).**

Use the following to cite our paper:
```bibtex
@inproceedings{lee2022personachatgen,
  title={PERSONACHATGEN: Generating Personalized Dialogues using GPT-3},
  author={Lee, Young-Jun and Lim, Chae-Gyun and Choi, Yunsu and Lm, Ji-Hui and Choi, Ho-Jin},
  booktitle={Proceedings of the 1st Workshop on Customized Chat Grounding Persona and Knowledge},
  pages={29--48},
  year={2022}
}
```

## 🎭 PersonaChatGen

You can now download PersonaChatGen dataset from the [google drive](https://drive.google.com/drive/folders/1-q2ZrnYpVzLB17rm9Net4UOLpE54dSaR?usp=sharing).
We provide the train and validation sets following the format of the original [PersonaChat](https://arxiv.org/abs/1801.07243) dataset, as provided by the [ParlAI](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/personachat) framework.

## 🤖 How to make PersonaChatGen using GPT-3?

To construct the PersonaChatGen dataset using GPT-3, we propose a pipeline consisting of three stages: (1) ProflieGen Creation, (2) Persona Set Creation, and (3) PersonaChatGen Creation. The detailed information is in our paper. Please follow the below instruction step-by-step.

### Preparation

#### Installation

Install the required set of libraries as follows:
```
pip install -r requirements.txt
```

#### Set up OpenAI API Key

Set up the OpenAI API Key in the function of `get_response()` in `prompt_generator.py` as follows:

```python
openai.api_key = "<API_KEY>"
openai.organization = "<ORG_ID>"
```

### ProfileGen Creation

#### Generation
Run the command below to generate various profile-related sentences using GPT-3.

```python
python profile_main.py
```

#### Filtering
Run the command below to filter low-quality sentences based on regex-based filtering, exact matching persona entity, preserving persona category, and duplication filtering.

```python
python profile_filtering.py
```

### Persona Set Creation

Run the command below to create persona sets using our proposed simple algorithm, namely CoNL (Contradiction-based Iterative Sentence Replacement).
> 🚨 Please note that this algorithm and the accompanying implementation can take a significant amount of time to create numerous persona sets. We encourage other contributors to improve it for greater efficiency.

```python
python conl_main.py
```

### PersonaChatGen Creation

#### Generation
Run the command below to generate PersonaChatGen dataset using GPT-3.

```python
python chat_main.py
```

#### Filtering
Run the command below to filter low-quality dialogues based on copy-paste, persona consistency, toxicity filtering.

```python
python chat_filtering.py
```

## Have any question?

Please contact [Young-Jun Lee](https://sites.google.com/view/passing2961/%ED%99%88) at yj2961@kaist.ac.kr or passing2961@gmail.com.

