# Understanding the Transformers Architecture

<img width="481" alt="image" src="https://github.com/bala1802/Neural-Networks-and-Deep-Learning/assets/22103095/bb2e199d-de8c-4dca-b75e-d80ee6567098">

Note:
  - This blog is written to understand what happens internally in the Transformers code.
  - I have used one simple example where I have a Source Sentence, represented in Tensor as `tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0]])`
  - and a Target Sentence, represented in Tensor as `tensor([[1, 7, 4, 3, 5, 9, 2, 0]])`
  - The Input and Output are passed to the Encoder, Decoder  blocks to understand how the data is handled, what will be the output and shape of the tensors in each stage

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

