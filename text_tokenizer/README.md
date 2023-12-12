# CLIP Tokenizer

The `custom_op_cliptok.onnx` file is a single op ONNX model which implements a CLIP style tokenizer (i.e. a byte level
Byte-Pair encoding tokenizer). It is generated from the vocabulary and merge list by
[onnxruntime-extensions](https://github.com/microsoft/onnxruntime-extensions), and contains the single operation
`ai.onnx.contrib.CLIPTokenizer`. It can be swapped out for another tokenizer if necessary by regenerating the ONNX file
using a different vocabulary and/or merge list.

## Provenance

This file is taken from the C# implementation of stable diffusion inference [commit 0323ce3](https://github.com/cassiebreviu/StableDiffusion/commit/0323ce34c1b490f539b5ae6244e52b8706a6dba7),
specifically [this file](https://github.com/cassiebreviu/StableDiffusion/blob/0323ce34c1b490f539b5ae6244e52b8706a6dba7/StableDiffusion.ML.OnnxRuntime/cliptokenizer.onnx).

## License and Copyright

MIT License

Copyright (c) 2023 Cassie Breviu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.