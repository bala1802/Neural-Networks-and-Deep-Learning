# Understanding the Transformers Architecture

<img width="481" alt="image" src="https://github.com/bala1802/Neural-Networks-and-Deep-Learning/assets/22103095/bb2e199d-de8c-4dca-b75e-d80ee6567098">

Note:
  - This blog is written to understand what happens internally in the Transformers code.
  - I have used one simple example where I have a Source Sentence, represented in Tensor as `tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0]])`
  - and a Target Sentence, represented in Tensor as `tensor([[1, 7, 4, 3, 5, 9, 2, 0]])`
  - The Input and Output are passed to the Encoder, Decoder blocks to understand how the data is handled, what will be the output and shape of the tensors in each stage

## Data Understanding:

- Shape of the Source Tensor - `torch.Size([1, 9])`
  * Batch Size is `1`. We just have only one input sentence
  * The length of the input sequence is `9`
- Shape of the Target Tensor - `torch.Size([1, 8])`
  * Batch Size is `1`. We have one corresponding target tensor
  * The length of the target sentence is `8`
- The Padding Index will be `0` while padding the input and output sentence
- The Source and Target Vocabulary size is `10`

## Encoder:

<img width="245" alt="image" src="https://github.com/bala1802/Neural-Networks-and-Deep-Learning/assets/22103095/92153333-6931-46d5-97e2-af32449370ea">

### Preparing the Input:

`src` and `src_mask`
- `src` -> `tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0]])`
- Shape of the `src` -> `torch.Size([1, 9])`
- Let's understand the `src_mask`
  * Inside a batch, the input sequence length will vary for each source sentence. The Sequence length is determined to be the maximum length of the sentence available in the training dataset. So, for the rest of the training dataset, the (max_length - len(current_sentence) is set to be masked.
  * For the `src_mask` in our example will be `tensor([[[[ True,  True,  True,  True,  True,  True,  True,  True, False]]]])`.
  * The value `0` denotes that it is a padded token. So `False` as a last element in the `src_mask`

### Encoder Initialization:
- The `embedding_size` is 256
- The `word_embedding` dimension is configured to `(src_vocab_size, embedding_size)` = `(10, 256)`. The Source sentence's vocabulary size is `10`
- The `position_embedding` dimension is configured to `(max_sequence_length, embedding_size)` = `(9, 256)`. The Sequence length of the source sentence is `9`

Now we're ready with the `src` and `src_mask`, lets understand what the `Encoder` does internally
- As a first step, the `position_embeddings` and `word_embeddings` are calculated
- We have the sequence length as `9`. And the batch size as `1`. So the `position_embeddings` shape is initialized to `(1,9)` Later Calculated
- After passing the `source` statement to the `word embedding` layer, the `word_embeddings` dimension is resulted as `torch.Size([1, 9, 256])`
- After passing the `position` to the `position embedding` layer, the `position_embeddings` dimension is resulted as `torch.Size([1, 9, 256])`
- The `word_embedding` and `position_embedding` are added, `out = result_of_word_embedding + result_of_position_embedding`.
- `out` is the `value`, `key` and `query` to the `Transformer Block`  

### Encoder Calling the `Transformer Block`

The `out` which will be the `value`, `key` and `query` sent to the `Transformer Block`. The `Encoder` is configured with `6` layers. It means the `Transformer Block` is called `6` times

Let's understand the `Transformer Block`

![Alt text](image.png)

The Transformer block is comprised of (sequentially), 
  A. Multihead Attention layer
  B. Add & Norm
  C. Feed Forward layer
  D. Add & Norm

  #### A. Multihead Attention Layer:
  - The `key`, `value` and `query` are sent into the `Self Attention` layer
  - `key`, `value`, `query` = `out`. Shape is `(1,9,256)`
  - From the above, we can understand that, the `Self Attention` takes the `Batch Size` - `1`; Sequence length of the input source `9`; The embedding dimension is `256` 
  - `Self Attention` Initialization:
    * The number of `heads` are configured in the beginning as `8`
    * The `embed` dimension is set to `256`
    * Calculate the dimension for each `head`; `dimension_of_each_head` is `(256/8) = 32`
    * A `Linear` layer is initialized for `value`, `key` and `query`
    * `Linear Layer for value` - with the shape of `(dimension_of_each_head, dimension_of_each_head)` -> `(32, 32)`
    * `Linear Layer for key` - with the shape of `(dimension_of_each_head, dimension_of_each_head)` -> `(32, 32)`
    * `Linear Layer for Fully Connected Output` - with the shape of `(dimension_of_each_head*number_of_heads, dimension_of_each_head)` -> `(32*8, 256)` -> `(256, 256)`