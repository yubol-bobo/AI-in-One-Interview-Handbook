<a id="transformer"></a>
# [To be reviewed] Transformer: Architecture and Computation

[Handbook contents](../README.md#table-of-contents) | [Previous: Deep Learning](deep-learning.md) | [Next: NLP (Pre-LLM Era)](nlp.md)

## Contents

- [1. Tensor Shapes and the Residual Stream](#tf-shapes)
- [2. Token Embeddings: IDs to a Learned Coordinate System](#tf-embedding)
- [3. Position Information: Addition, Relative Bias, and RoPE](#tf-position)
- [4. Q, K, V and Scaled Dot-Product Attention](#tf-attention)
- [5. Visibility Masks, Padding, and Teacher Forcing](#tf-masking)
- [6. Multi-Head Attention, MQA, and GQA](#tf-heads)
- [7. Encoder Memory and Decoder Cross-Attention](#tf-cross-attention)
- [8. Positionwise MLP, SwiGLU, and MoE](#tf-mlp)
- [9. Residual Paths, LayerNorm, RMSNorm, and Dropout](#tf-norm)
- [10. Vocabulary Projection, Loss, and Decoding](#tf-readout)
- [11. Prefill, Incremental Decode, and KV Caching](#tf-cache)
- [12. Parameter, Arithmetic, and Memory Costs](#tf-cost)
- [13. Architecture Families and Historical Examples](#tf-architecture-families)
- [14. Minimal Attention Reference and Failure Checks](#tf-reference)


The diagram separates the **original encoder–decoder Transformer** from one **common modern decoder-only design**. The original uses Post-LayerNorm and a decoder cross-attention sublayer; the illustrated modern block uses Pre-RMSNorm, RoPE, grouped-query attention, and SwiGLU. These are specific configurations, not features required of every Transformer. Read the data flow from bottom to top; each dashed enclosure is a repeated layer, while embeddings and output heads sit outside that repetition.

![Transformer architecture: original encoder–decoder and a common modern decoder-only block](../figs/transformer-architecture.png)

[Editable SVG](../figs/transformer-architecture.svg). This original schematic omits dropout, projection biases, and implementation-specific distributed layouts for readability. The following sections specify the operations and assumptions behind each box.

<a id="tf-shapes"></a>
## [To be reviewed] 1. Tensor Shapes and the Residual Stream

Use batch size $`B`$, source length $`S`$, target length $`T`$, model width $`d`$, query-head count $`h_q`$, KV-head count $`h_{kv}`$, head width $`d_h`$, MLP width $`m`$, vocabulary size $`V`$, and layer count $`L`$. In the common head layout, $`h_qd_h=d`$; other parameterizations are possible. The residual stream keeps shape $`[B,T,d]`$ through a decoder layer, permitting elementwise skip additions. It is a changing contextual representation, not a table of independently meaningful neuron labels.

| Quantity | Common shape | Role |
| --- | --- | --- |
| Token IDs | [B, T] | Integer indices, not floating-point hidden states |
| Residual stream | [B, T, d] | Shared interface between attention and MLP sublayers |
| Q | [B, h_q, T, d_h] | Queries at target/query positions |
| K / V in self-attention | [B, h_kv, T, d_h] | Keys and content at permitted sequence positions |
| Attention scores, MHA | [B, h_q, T, T] | Query-to-key comparisons before softmax |
| Encoder memory | [B, S, d] | Source states supplied to decoder cross-attention |
| Vocabulary logits | [B, T, V] | Unnormalized next-token scores |

**Shape check:** in cross-attention, query length is T and key/value length is S, so scores are [B, h_q, T, S]. Confusing those axes can create a numerically valid but semantically wrong softmax. Normalize over **keys for each query**, not over the query dimension. [MultiheadAttention shape reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)

<a id="tf-embedding"></a>
## [To be reviewed] 2. Token Embeddings: IDs to a Learned Coordinate System

An embedding matrix $`E\in\mathbb{R}^{V\times d}`$ maps token ID $`x_t`$ to row $`E[x_t]`$. A lookup is equivalent to multiplying a one-hot token vector by E without materializing that vector. Repeated occurrences share input parameters; their **contextual hidden states** can diverge after attention. The embedding is trained through the downstream objective, not by assigning a fixed semantic definition to each coordinate. [Embedding API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

**Example:** “bank” near “river” and “bank” near “loan” may begin with the same token embedding but acquire different contextual states. A rare identifier can be split across several subwords; tokenization determines both the alphabet and the amount of context consumed. See [tokenization](large-language-models.md#tokenization) and [retrieval representations](large-language-models.md#embedding) for the separate question of obtaining a useful sentence-level vector.

The original Transformer multiplies token embeddings by $`\sqrt d`$ before adding sinusoidal positions. Many modern models use another scaling convention. Match the trained model; adding this factor to an arbitrary checkpoint changes activation scales. A tied output head can share E with the vocabulary projection, reducing parameter storage while coupling input/output learning. [Original architecture](https://arxiv.org/abs/1706.03762), [weight tying](https://arxiv.org/abs/1608.05859)

<a id="tf-position"></a>
## [To be reviewed] 3. Position Information: Addition, Relative Bias, and RoPE

Without position-dependent information or masks, self-attention is permutation-equivariant: permuting input positions permutes the outputs. This is different from a permutation-invariant pooled representation. A causal mask already introduces order constraints; positional mechanisms supply additional position/distance structure.

**Absolute addition.** A learned table supplies $`P_t`$ and the layer input becomes $`E[x_t]+P_t`$. A fixed sinusoidal alternative uses $`P_{t,2i}=\sin(t/10000^{2i/d})`$ and $`P_{t,2i+1}=\cos(t/10000^{2i/d})`$. A learned lookup has a configured position range; a computable sinusoid can be evaluated beyond training lengths, but that mathematical extension does not establish useful extrapolation. [Original positional encoding](https://arxiv.org/abs/1706.03762)

**Relative score bias.** Add a position-dependent quantity $`b_{ij}`$ to the attention score for query i and key j. T5 uses learned distance buckets; ALiBi uses head-specific linear distance penalties. This changes comparison scores rather than simply adding a vector to token inputs. Bucketing sacrifices some distance resolution, and a bias's formula alone does not guarantee long-context retrieval. [T5](https://jmlr.org/papers/v21/20-074.html), [ALiBi](https://arxiv.org/abs/2108.12409)

**RoPE.** Apply position-dependent rotations to query/key coordinate pairs: $`q_i'=R_iq_i`$, $`k_j'=R_jk_j`$. For a shared frequency system, $`(q_i')^Tk_j'=q_i^TR_i^TR_jk_j=q_i^TR_{j-i}k_j`$. The dot product therefore contains relative-position structure even though rotations use absolute position indices. A rotation preserves each pair's Euclidean norm; it does not make all attention scores depend only on distance because content vectors also matter. Values are commonly left unrotated. [RoFormer](https://arxiv.org/abs/2104.09864)

**Implementation example:** if a cached key was already rotated at absolute position 127, appending token 128 should not rotate that old key again. Preserve the position convention across prefill, cache reuse, padding, and sequence packing. Partial-RoPE dimensions, frequency scaling, and long-context adaptation are checkpoint-specific. Recurrent/ODE-generated encodings and multiplicative input encodings are additional families, rather than interchangeable names for RoPE. Validate changed position mechanisms using length and evidence-position sweeps, as discussed in [mid-training](large-language-models.md#mid-training).

<a id="tf-attention"></a>
## [To be reviewed] 4. Q, K, V and Scaled Dot-Product Attention

For a self-attention input X, the projected matrices are $`Q=XW_Q`$, $`K=XW_K`$, and $`V=XW_V`$. After splitting heads, one head computes $`A=\operatorname{softmax}(QK^T/\sqrt{d_h}+M)`$ and $`O=AV`$. Q determines what the current position compares, K determines how candidate positions are addressed, and V supplies the content being mixed. Key/value roles need not coincide with literal words or human-defined facts. [Scaled dot-product implementation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

Under independent, zero-mean, unit-variance Q/K coordinates, a dot product's variance is $`d_h`$. Division by $`\sqrt{d_h}`$ controls the initial scale before softmax. For probability vector a, the softmax Jacobian is $`\operatorname{diag}(a)-aa^T`$; highly saturated probabilities can make changes in some logits have little effect. Learned dependence and normalization alter the simple variance assumptions, so this is a motivating calculation rather than a universal initialization theorem.

**Numerical example:** scaled allowed logits [ln 3, 0] produce weights [0.75, 0.25]. With value vectors (1, 0) and (0, 2), the output is (0.75, 0.5). The **comparison scores** choose the mixture, while the **values** determine what is mixed. Changing only V changes the output without changing those attention weights. This also explains why a heatmap is not by itself a causal explanation of the model's final prediction; later transformations and residual paths remain active. See [attention-based explanation](machine-learning.md#attention-based).

<a id="tf-masking"></a>
## [To be reviewed] 5. Visibility Masks, Padding, and Teacher Forcing

A causal query at position i may attend only to permitted keys at positions $`j\leq i`$. An additive mask contributes zero for allowed positions and a suppressing value, conceptually negative infinity, for blocked ones. Padding masks remove invalid **keys**; loss masks independently choose which target tokens contribute to training. Packed examples may also require document/segment boundaries so one example cannot read another.

**Teacher forcing** supplies known input tokens at all positions, so a causal training pass can compute many next-token predictions in parallel. It does not reveal the target being predicted: the logits at input position t predict the next token at t+1 under the usual shifted-label convention. A correct mask and a correct label shift solve different leakage problems. The [pre-training loss example](large-language-models.md#causal-loss-example) shows the shift explicitly.

**Cached decode pitfall:** for a query appended after P cached tokens, its absolute index is P, not zero. A generic lower-triangular mask built as if query/key axes both started at zero can incorrectly block valid history in rectangular attention. Derive visibility from the actual position indices. Avoid rows with no allowed key; softmax of all negative infinities is undefined in the reference formulation. Library conventions differ: PyTorch SDPA boolean masks use **True for allowed positions**, while some other attention APIs use True for blocked entries. [SDPA mask semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

<a id="tf-heads"></a>
## [To be reviewed] 6. Multi-Head Attention, MQA, and GQA

MHA computes several head outputs, concatenates them, and applies an output projection: $`\operatorname{MHA}(X)=\operatorname{Concat}(O_1,\ldots,O_{h_q})W_O`$. W_O mixes head coordinates back into the residual width d; concatenation alone is not the complete sublayer. With ordinary square MHA projections and no biases, Q/K/V/O contain approximately $`4d^2`$ weights. Multiple heads enable different learned addressing/content subspaces, but they do not guarantee a clean human-readable role per head.

| Design | Query heads | KV heads | Main structural change |
| --- | --- | --- | --- |
| MHA | h_q | h_q | Distinct K/V per query head |
| MQA | h_q | 1 | One shared K/V head |
| GQA | h_q | h_kv between 1 and h_q | Groups of query heads share K/V |

For the common grouped layout, h_q is divisible by h_kv. Q retains all query heads, while K/V projections and cache shrink. **Example:** 32 query heads with 8 KV heads share each KV head among four query heads. Relative to 32-KV-head MHA, that cache has one quarter the K/V entries at the same layer count, length, dtype, and head width. It does not imply a fourfold speedup: attention computation, other weights, batching, and hardware still matter. The GQA paper also studies converting existing MHA checkpoints by additional training; changing head counts in a saved checkpoint is not a free tensor-layout operation. [GQA](https://arxiv.org/abs/2305.13245)

<a id="tf-cross-attention"></a>
## [To be reviewed] 7. Encoder Memory and Decoder Cross-Attention

In the original encoder–decoder model, encoder self-attention processes the source bidirectionally, while decoder self-attention is causal over the target. Cross-attention takes queries from decoder states and keys/values from encoder memory: $`Q=H_{\mathrm{dec}}W_Q`$, $`K=H_{\mathrm{enc}}W_K`$, $`V=H_{\mathrm{enc}}W_V`$. Its source dimension is S rather than target length T. The source need not be causally masked if it is fully provided; source padding still matters. [Encoder/decoder reference implementation](https://nlp.seas.harvard.edu/annotated-transformer/)

**Translation example:** while generating a target verb, the decoder can select source positions that appeared anywhere in the supplied sentence. Its decoder Q changes as the target prefix grows, but a layer's encoder-derived K/V can be projected once and reused while the source is unchanged. Each decoder layer may have its own cross-attention projections. This differs from a single shared bidirectional prefix stack, whose representations and parameter-sharing pattern are different.

A conventional decoder-only block has no separate encoder-memory cross-attention. Prompt tokens and generated tokens occupy the causal stream together. Multimodal or retrieval systems may insert extra cross-attention or other fusion modules; the presence of external information does not determine a unique architecture.

<a id="tf-mlp"></a>
## [To be reviewed] 8. Positionwise MLP, SwiGLU, and MoE

A standard MLP transforms each position independently, $`\operatorname{FFN}(x)=\phi(xW_1+b_1)W_2+b_2`$, with widths $`d\to m\to d`$. Its shared weights perform nonlinear feature computation at every token; the MLP itself does not directly mix sequence positions. Attention supplies context-dependent token mixing, and repeated attention/MLP composition changes both addressing and representations. Viewing FFNs as learned key/value memories can be a useful mechanistic hypothesis, not an exact semantic interpretation of every neuron. [Feed-forward analysis](https://arxiv.org/abs/2012.14913)

One common bias-free SwiGLU form is $`\operatorname{FFN}(x)=[\operatorname{SiLU}(xW_g)\odot(xW_u)]W_d`$. The gate/up matrices map d to m and the down matrix maps m to d. Its matrix weight count is $`3dm`$, compared with $`2dm`$ for a two-matrix MLP at the **same** m. Consequently, replacing a 4d-width ordinary FFN with a gated FFN often adjusts m toward $`8d/3`$ for similar parameter count, rounded to implementation-friendly sizes; the ratio is a budget calculation, not a universal optimal width. [GLU variants](https://arxiv.org/abs/2002.05202)

**Example:** at d = 4,096, a 4d ordinary MLP has about 134.2 million matrix weights; a gated MLP with that same hidden width has about 201.3 million. Reporting equal hidden width does not establish equal model capacity or compute.

MoE replaces a dense FFN with a router and several expert FFNs, often activating a small selected subset per token. Total parameter storage, activated computation, routing communication, and load balance are separate quantities. A top-2 router does not mean all experts fit into memory at the cost of two experts. See [expert parallelism](large-language-models.md#distributed-training) for training layout. [Switch Transformers](https://jmlr.org/papers/v23/21-0998.html)

<a id="tf-norm"></a>
## [To be reviewed] 9. Residual Paths, LayerNorm, RMSNorm, and Dropout

The original Post-LN sublayer is $`x'=\operatorname{LN}(x+F(x))`$, omitting dropout in this notation. A common Pre-LN/RMSNorm sublayer is $`x'=x+F(\operatorname{Norm}(x))`$, usually followed by a final normalization outside the repeated decoder stack. Residual addition requires compatible shapes; neither attention nor the MLP can silently change the residual width.

For a simplified Pre-LN block, the Jacobian contains a direct identity path: $`J=I+J_FJ_{\mathrm{Norm}}`$. Post-LN instead has $`J=J_{\mathrm{Norm}}(I+J_F)`$. This helps explain different initialization/optimization behavior, but cancellation, depth, scale, and the full architecture still matter. Neither formula proves gradients are always stable or that one placement always yields better final quality. [Normalization placement](https://arxiv.org/abs/2002.04745)

LayerNorm typically normalizes **hidden features within each token** using a mean and variance. A common RMSNorm computes $`\operatorname{RMSNorm}(x)=g\odot x/\sqrt{d^{-1}\sum_i x_i^2+\epsilon}`$, omitting mean subtraction and usually an additive bias. Epsilon, reductions, dtype, and affine settings affect numerical behavior. RMSNorm is not normalization across batch or sequence length. [RMSNorm](https://arxiv.org/abs/1910.07467)

Dropout may act on attention probabilities, sublayer outputs, embeddings, or residual branches depending on the recipe; these placements are different regularizers. Training/inference settings must match the implementation. Some modern LLMs use little or no dropout in particular configurations. The diagram omits it so that the two normalization orders remain legible; production code must specify it explicitly.

<a id="tf-readout"></a>
## [To be reviewed] 10. Vocabulary Projection, Loss, and Decoding

After the final hidden state, a vocabulary head produces $`z_t=h_tW_{\mathrm{vocab}}+b`$ with $`W_{\mathrm{vocab}}\in\mathbb{R}^{d\times V}`$. Weight tying may use $`W_{\mathrm{vocab}}=E^T`$. Logits are scores, while softmax converts them to a categorical distribution. Fused cross-entropy commonly consumes logits directly; applying softmax first and then passing probabilities to a logits-based loss changes the computation. [CrossEntropyLoss API](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

For the input `[BOS, A, B]`, the three aligned next-token labels can be `[A, B, EOS]`. Training on the current input token at the same position instead would teach a different, often trivial copying task if the token is already visible. In a chat example, the causal mask still protects visibility while a separate loss mask may exclude system/user targets. Padding and masked targets must be excluded from the denominator of the desired mean.

At generation time, select from the final available position's logits using the specified policy: greedy selection, temperature scaling, top-k, top-p, or another rule. Selecting a token, appending it, and updating the cache defines the next state. Temperature zero is a greedy limiting convention, not division by zero. Sampling parameters affect outputs, measured benchmark scores, and the rollout distribution used in [RL/OPD](large-language-models.md#sft-rl-opd-comparison).

<a id="tf-cache"></a>
## [To be reviewed] 11. Prefill, Incremental Decode, and KV Caching

**Prefill** computes states for a known prompt, usually in parallel over positions under the chosen mask. **Decode** computes new-token queries/keys/values and attends to cached earlier keys/values. Earlier causal hidden states are unchanged by appending a future token, so their K/V do not need to be recomputed. K/V are layer-specific; the cache is not just a copy of the input embeddings. [Cache conventions](https://huggingface.co/docs/transformers/cache_explanation)

For an attention-only step with a cache of length P, a common layout has Q [B, h_q, 1, d_h] and K/V [B, h_kv, P+1, d_h]. New keys must use the correct positional transformation before storage. Encoder–decoder generation can additionally reuse fixed source cross-attention K/V. A bidirectional prefix may be cached for unchanged-prefix continuation, but modifying that prefix can require recomputation.

**What caching buys:** a new token attends to about P previous positions rather than recomputing every earlier token's layer projections. It does not eliminate attention's growing history reads or the sequential dependence of exact ordinary autoregressive generation. Cache memory grows with active tokens and concurrency; [LLM inference](large-language-models.md#llm-inference) gives the memory formula, a 1-GiB-per-sequence example, paging, and speculative decoding.

**Correctness experiment:** compare full causal attention with incremental cached attention using identical projected Q/K/V, positions, weights, and disabled dropout. Then change a future input token and verify that earlier causal outputs stay unchanged. Compare within appropriate floating-point tolerances; speed or a plausible generated sentence is not a cache-correctness test.

<a id="tf-cost"></a>
## [To be reviewed] 12. Parameter, Arithmetic, and Memory Costs

For ordinary MHA and a two-matrix FFN, one layer has approximately $`4d^2+2dm`$ matrix weights; with m = 4d this is $`12d^2`$, excluding norms, biases, embeddings, and the vocabulary head. A decoder layer with an additional ordinary cross-attention sublayer adds roughly another $`4d^2`$ at equal source/target widths. GQA and gated MLPs require their actual projection sizes; do not reuse this coefficient indiscriminately.

In **multiply-accumulate operations (MACs)**, ordinary dense MHA prefill uses roughly $`4BTd^2`$ for Q/K/V/O projections plus $`2BT^2d`$ for QK and AV; an ordinary FFN adds $`2BTdm`$. One MAC is often counted as two FLOPs. Masks, causal kernels, precision, and implementation details change constants. Small-T workloads can be projection/MLP dominated; long sequences increase attention's relative cost. These are arithmetic estimates, not wall-clock latency predictions.

**Memory example:** a single explicitly materialized score matrix at B = 1, h = 32, T = 8,192 and two bytes per score occupies 4 GiB. That is one tensor, not the full training peak. FlashAttention tiles operations and uses online softmax/recomputation to reduce memory traffic and avoid storing the full score matrix in high-bandwidth memory. Its dense exact attention still has quadratic sequence-length arithmetic, up to kernel/mask constants. “Exact” means the attention operator rather than a sparse/low-rank approximation; floating-point reduction order can differ. [FlashAttention](https://arxiv.org/abs/2205.14135)

Decode performance often depends on weight/cache bandwidth, batch size, and concurrency as well as arithmetic. Report time to first token, inter-token latency, throughput, and peak memory under specified prompt/output distributions. A fourfold reduction in one attention tensor does not imply a fourfold reduction in total model cost.

<a id="tf-architecture-families"></a>
## [To be reviewed] 13. Architecture Families and Historical Examples

| Family | Visibility / modules | Typical objective and task fit |
| --- | --- | --- |
| Encoder-only, e.g. BERT/RoBERTa | Bidirectional self-attention; no causal generation head required | Masked-token representations, tagging/classification/extractive QA |
| Encoder–decoder, e.g. original Transformer/T5/BART | Source encoder, causal decoder, source cross-attention | Conditional generation such as translation or summarization; objectives vary |
| Decoder-only, e.g. GPT-style/Llama-style models | Causal stream; commonly no separate source cross-attention | Continuation, instruction generation, tool-use trajectories |
| Single-stack prefix LM | Bidirectional supplied prefix plus causal continuation | Prefix-conditioned generation with a specified target-loss mask |

BERT uses MLM and the original NSP objective; RoBERTa changes the recipe, including removal of NSP, more training/data, and dynamic masking. XLNet's permutation-factorization objective and two-stream attention are another historical design, rather than a universal improvement over a causal decoder. T5 uses a pre-normalization RMS-style variant despite the original 2017 figure's Post-LN blocks. Compare architectures with specified objectives/data and compute. [BERT](https://aclanthology.org/N19-1423/), [RoBERTa](https://arxiv.org/abs/1907.11692), [XLNet](https://arxiv.org/abs/1906.08237), [T5](https://jmlr.org/papers/v21/20-074.html)

<a id="tf-reference"></a>
## [To be reviewed] 14. Minimal Attention Reference and Failure Checks

This educational implementation exposes score/mask/softmax behavior; it materializes scores and is not a replacement for an optimized attention kernel. Inputs already contain projected heads and any required positional transform. The function supports matching head counts; GQA requires an explicit grouping layout or a supported kernel.

```python
import math
import torch

def reference_attention(q, k, v, allowed=None):
    # q: [B, H, Tq, Dh]; k/v: [B, H, Tk, Dh]. True means ALLOWED.
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if allowed is not None:
        if allowed.dtype != torch.bool:
            raise TypeError("allowed must be a boolean mask")
        if not allowed.any(dim=-1).all():
            raise ValueError("Each query must have at least one allowed key")
        scores = scores.masked_fill(~allowed, float("-inf"))
    softmax_dtype = (torch.float32 if scores.dtype in
                     (torch.float16, torch.bfloat16) else scores.dtype)
    weights = torch.softmax(scores, dim=-1, dtype=softmax_dtype).to(v.dtype)
    return weights @ v, weights

def causal_allowed(query_length, key_length, past_length=0, device=None):
    # Append-only self-attention with query indices starting after the cache.
    if key_length != past_length + query_length:
        raise ValueError("Expected cached prefix plus current query block")
    q_positions = past_length + torch.arange(query_length, device=device)
    k_positions = torch.arange(key_length, device=device)
    return k_positions[None, :] <= q_positions[:, None]
```

**Test the mechanisms separately:** (1) changing future K/V must not change past causal outputs; (2) cached and full causal results should agree; (3) masked-key probability should be zero under the chosen reference formulation; (4) unmasked rows should sum to one before attention dropout; (5) RoPE should preserve rotated-pair norms and relative-angle identities; (6) label-shift and padding tests should compare the actual valid target set. Fused implementations can differ in precision and mask conventions, so establish the intended operator first.

Original study resources: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), [original lecture reference](https://www.youtube.com/watch?v=nzqlFIcCSWQ). For optimization and distributed execution, continue with [shared training systems](large-language-models.md#llm-engineering).
