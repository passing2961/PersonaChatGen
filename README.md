# 🎭 PersonaChatGen

**This is the official github repository for [PERSONACHATGEN: Generating Personalized Dialogues using GPT-3](https://aclanthology.org/2022.ccgpk-1.4/).**

- **TL;DR**: Recently, many prior works have made their own agents generate more personalized and engaging responses using personachat. However, since this dataset is frozen in 2018, the dialogue agents trained on this dataset would not know how to interact with a human who loves “Wandavision.” One way to alleviate this problem is to create a large-scale dataset. In this work, we introduce the pipeline of creating personachatgen, which is comprised of three main components: Creating (1) profilegen, (2) Persona Set, and (3) personachatgen. To encourage GPT-3’s generation ability, we also defined a taxonomy of hierarchical persona category derived from social profiling taxonomy. To create the speaker consistent persona set, we propose a simple contradiction-based iterative sentence replacement algorithm, named CoNL. Moreover, to prevent GPT-3 generating harmful content, we presented two filtering pipelines, one each for profilegen and personachatgen. Through analyzing of personachatgen, we showed that GPT-3 can generate personalized dialogue containing diverse persona. Furthermore, we revealed a state-of-the-art Blender 90M trained on our dataset that leads to higher performance.

📜 [**Slide**](https://drive.google.com/file/d/1FCtRyoxySbeHORUt9brhhzhVrULzAlZ3/view?usp=share_link)

**🏆 PersonaChatGen won the Best Paper Award at CCGPK@COLING 2022!**

## Reference
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

## 🔎 ProfileGen

You can now download ProfileGen dataset from the [google drive](https://drive.google.com/drive/folders/18a6tBapA3IqMyfjxL0z2MQbTNwMjhxOf?usp=share_link).
We provide individual json files, where each file is related to the persona category. (See Table 15, 16, 17 in our paper)
Each file contains a list of profile-related sentence generated by GPT-3, where each element in the list consists of `sentence`, `attr`, `value`, and `nli_score`. Please check a sample data in `dataset/profile_sample.json`.


## 🎭 PersonaChatGen

You can now download PersonaChatGen dataset from the [google drive](https://drive.google.com/drive/folders/146MR4ODZ51eesK17R6LorrG9bzX-zxPs?usp=share_link).
We provide the train and validation sets following the format of the original [PersonaChat](https://arxiv.org/abs/1801.07243) dataset, as provided by the [ParlAI](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/personachat) framework.
Please check a sample data in `dataset/chat_sample.txt`.

## 🤖 How to make PersonaChatGen using GPT-3?

To construct the PersonaChatGen dataset using GPT-3, we propose a pipeline consisting of three stages: **(1) ProflieGen Creation, (2) Persona Set Creation, and (3) PersonaChatGen Creation**. The detailed information is in our paper. Please follow the below instruction step-by-step.

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

## Acknowledgements

This work was supported by the KT Corporation. We thank all KT researchers for helpful discussions.

## Have any question?

Please contact [Young-Jun Lee](https://sites.google.com/view/passing2961/%ED%99%88) at yj2961@kaist.ac.kr or passing2961@gmail.com.

## License

This repository is MIT licensed. See the [LICENSE](https://github.com/passing2961/PersonaChatGen/blob/main/LICENSE) file for details.
