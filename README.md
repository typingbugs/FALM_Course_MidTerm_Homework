# 《大模型基础与应用》课程中期作业

- 姓名：柯劲帆
- 学号：25120323

---

本项目fork了🤗Transformers官方仓库，于 `src/transformers/models/my_model` 中自定义了模型代码。

项目结构：
- `src/transformers/`：🤗Transformers源码
    - `src/transformers/models/my_model/`：我的模型源码
- `data/`：各种数据
    - `data/iwslt2017-en-de/`：训练数据jsonl文件
    - `data/model/`：从头训练的模型配置、分词器文件
    - `data/results/`：测试结果
- `train/`：训练代码
- `test/`：测试代码
- `train_configs/`：训练配置文件
- `scripts/`：启动脚本
- `results/`：实验结果
- `requirements.txt`：pip环境配置

## 运行方法
1. 配置环境：

    ```shell
    pip install -e ".[torch]"
    pip install -r requirements.txt
    ```

2. 训练：
    ```shell
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    bash scripts/train.sh
    ```
    训练模型需要1张或以上RTX 3090（24G）；使用4张3090训练时间约为2.5小时。

    如有需要，可以：
    - 在 `data/model/config.json` 中配置模型超参数，放入tokenizer文件；
    - 在 `train_configs/train.yaml` 中配置训练超参数，如学习率、随机种子等；

3. 测试：
    ```shell
    bash scripts/test.sh
    ```