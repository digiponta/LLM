# 自作LLM / Homemade LLM

Pythonだけで、LLMの上位層だけでなく **Tensor・仮想GPU・命令セット・スケジューラ・Transformer・自動微分・Optimizer** まで段階的に自作していく教育用プロジェクトです。

NumPyやPyTorchに依存せず、LLMの内部で何が起きているかを、できるだけ下位層から追える構成を目指しています。

## 全体構成

```text
gpu_memory.py
    ↓
gpu_isa.py
    ↓
gpu_compute.py
    ↓
gpu_scheduler.py
    ↓
tensor.py
    ├──────────────→ tokenizer.py
    ↓
embedding.py
    ↓
layers.py
    ↓
attention.py
    ↓
ffn.py
    ↓
transformer.py
    ↓
llm.py
    ↓
loss.py
    ↓
autograd.py
    ↓
optimizer.py
    ↓
train.py
```

## ファイル概要

### `gpu_memory.py`

仮想GPUのメモリモデルです。

主な機能:

- float値を格納するGPUメモリ
- メモリ確保 `allocate()`
- 解放 `free()`
- 読み書き `read()` / `write()`
- scalarアクセス
- メモリコピー
- メモリ内容のダンプ

現在はシンプルな線形アロケータで、解放領域の再利用はまだ実装していません。

---

### `gpu_isa.py`

自作仮想GPUの命令セット ISA を定義します。

主な命令:

- `LOAD`
- `STORE`
- `MOV`
- `ADD`
- `SUB`
- `MUL`
- `DIV`
- `FMA`
- `MATMUL`
- `SOFTMAX`
- `NOP`
- `HALT`

レジスタ、メモリ、即値オペランドと `GPUProgram` もここで定義しています。

---

### `gpu_compute.py`

`gpu_isa.py` で定義した命令を実際に実行する仮想GPU Compute Coreです。

主な機能:

- レジスタファイル
- Program Counter
- 算術命令実行
- Matrix Multiplication
- Softmax
- GPUProgram実行

`MATMUL` は教育目的でPythonの三重ループとして実装しています。

---

### `gpu_scheduler.py`

仮想GPUの並列実行モデルを表現します。

構造:

```text
Grid
 └─ Block
     └─ Warp
         └─ Thread
```

主な機能:

- Thread生成
- Warp単位実行
- Block構成
- Special RegisterによるThread ID管理
- Warp divergence検出

現時点では分岐命令は未実装です。

---

### `tensor.py`

仮想GPU上のメモリと演算を隠蔽するTensor抽象化です。

主な機能:

- 多次元Tensor
- shape管理
- element-wise演算
- Matrix Multiplication
- Softmax
- reshape
- transpose
- clone

NumPyは使用していません。

現在のelement-wise演算は教育目的の単純な実装であり、GPU Schedulerとの完全統合は今後の課題です。

---

### `tokenizer.py`

シンプルな文字単位Tokenizerです。

特殊Token:

- `<PAD>`
- `<UNK>`
- `<BOS>`
- `<EOS>`

主な機能:

- 語彙構築
- encode / decode
- batch encode
- padding / truncation
- JSON形式でのsave / load

---

### `embedding.py`

Token IDをベクトルへ変換するEmbedding層です。

```text
Token IDs
   ↓
Embedding Matrix
   ↓
[S, D]
```

現在の重みは `Parameter` として保持され、Optimizerから学習対象として取得できます。

---

### `layers.py`

ニューラルネットワークの基本Layerを実装しています。

含まれるLayer:

- `Linear`
- `ReLU`
- `GELU`
- `LayerNorm`
- `Softmax`
- `Sequential`

学習対象となる以下の値は `Parameter` 化されています。

- Linear weight
- Linear bias
- LayerNorm gamma
- LayerNorm beta

---

### `attention.py`

Single-Head Self-Attentionを実装します。

```text
X
├─ Linear → Q
├─ Linear → K
└─ Linear → V

Q @ K.T
   ↓
Scale
   ↓
Causal Mask
   ↓
Softmax
   ↓
Attention @ V
   ↓
Output Projection
```

Causal Attentionに対応しています。

Multi-Head Attentionは今後の拡張候補です。

---

### `ffn.py`

Transformer内部のFeed Forward Networkです。

```text
Input
  ↓
Linear
  ↓
GELU
  ↓
Linear
  ↓
Output
```

デフォルトではhidden dimensionを `4 * d_model` とする構成です。

---

### `transformer.py`

Pre-Norm型Transformer Blockと、そのStackを実装しています。

```text
x
│
├───────────────┐
↓               │
LayerNorm       │
↓               │
Attention       │
↓               │
Add ────────────┘
↓
LayerNorm
↓
FFN
↓
Add
↓
Output
```

複数Blockを積み重ねる `Transformer` クラスも含みます。

---

### `llm.py`

Embedding、Transformer、Vocabulary Projectionを接続した言語モデル本体です。

```text
Token IDs
   ↓
Embedding
   ↓
Transformer
   ↓
Linear
   ↓
Vocabulary Logits
```

主な機能:

- Forward inference
- Vocabulary logits
- Softmax probability
- Greedy next-token prediction
- 簡易token generation
- `parameters()` による全学習パラメータ取得

現時点では位置Embeddingは未実装です。

---

### `loss.py`

Language Model学習用のLossを実装します。

主な機能:

- Cross Entropy Loss
- `mean` / `sum` / `none` reduction
- next-token学習用token shift
- perplexity計算

例:

```text
[A, B, C, D]

Input : [A, B, C]
Target: [B, C, D]
```

---

### `autograd.py`

Reverse-Mode Automatic Differentiationの教育用実装です。

中心となるクラス:

- `AutoTensor`
- `Parameter`

対応演算:

- add
- subtract
- multiply
- scalar division
- matrix multiplication
- transpose
- sum
- mean
- exp
- log
- tanh

計算Graphを作成し、`backward()` によって勾配を逆伝播します。

---

### `optimizer.py`

自作Parameterを更新するOptimizerです。

実装済み:

- SGD
- Momentum SGD
- Adam
- Weight Decay
- `zero_grad()`
- `step()`

基本的な学習ループは次の形を想定しています。

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

### `train.py`

LLM学習用のTraining Loopの骨格です。

主な機能:

- Token sequenceから学習sample作成
- Context window生成
- Forward
- Cross Entropy Loss
- Perplexity表示
- Epoch loop
- Adam Optimizer生成

現時点では、LLM本体のForward計算がまだ完全な `AutoTensor` Graphになっていないため、`loss.backward()` と `optimizer.step()` は意図的に無効にしています。

## 現在できること

- 仮想GPUメモリを作る
- GPU命令を定義する
- 仮想GPU Compute Coreで命令を実行する
- Thread / Warp / Blockモデルを再現する
- Tensor演算を行う
- 文字Tokenizerを作る
- Embeddingを生成する
- Linear / GELU / LayerNorm / Softmaxを実行する
- Self-Attentionを実行する
- Transformerを構成する
- LLMのForward推論を行う
- Cross Entropy Lossを計算する
- AutoGrad単体でBackwardを実行する
- SGD / AdamでParameterを更新する
- LLM全体のTrainable Parameterを列挙する
- Training LoopのForward / Loss監視を行う

## 現在の重要な制約

LLM内部の各Forward処理は、まだ完全には `AutoTensor` ベースへ移行していません。

そのため現在は、

```text
LLM Forward
   ↓
Loss
```

までは実行できますが、

```text
LLM Forward
   ↓
Loss
   ↓
Backward
   ↓
全ParameterのGradient
   ↓
Optimizer
```

というEnd-to-End学習は未完成です。

## 次の開発ステップ

次の重点項目は **モデル全体のAutoTensor対応** です。

特に以下を順次対応する予定です。

1. `Linear` のAutoTensor対応
2. `Embedding` のGradient伝播
3. `GELU` のAutoTensor化
4. `LayerNorm` のBackward対応
5. `Softmax` のBackward対応
6. AttentionのAutoTensor化
7. Transformer全体のGraph接続
8. Cross Entropy LossのAutoTensor化
9. `loss.backward()` のEnd-to-End実行
10. `optimizer.step()` による実際のLLM学習

その後の拡張候補として、

- Positional Embedding
- Multi-Head Attention
- Mini-batch
- Broadcasting
- KV Cache
- Sampling / Temperature / Top-k / Top-p
- Model save / load
- GPU SchedulerとのTensor演算統合
- より低レベルなMATMUL ISA

などを想定しています。

## 実行方法

リポジトリをcloneします。

```bash
git clone https://github.com/digiponta/LLM.git
cd LLM
```

各ファイルには簡単なtest codeがあります。

例えば:

```bash
python gpu_memory.py
python gpu_compute.py
python tensor.py
python attention.py
python transformer.py
python llm.py
python autograd.py
python optimizer.py
python train.py
```

## プロジェクトの目的

このプロジェクトの目的は、高性能な既存LLM Frameworkを置き換えることではありません。

LLMを、

```text
Memory
↓
ISA
↓
Compute Core
↓
Scheduler
↓
Tensor
↓
Neural Network
↓
Transformer
↓
Language Model
↓
Loss
↓
Autograd
↓
Optimizer
↓
Training
```

という層に分解し、**LLMが計算機上でどのように成立しているのかを自分で実装しながら理解すること**を目的としています。
