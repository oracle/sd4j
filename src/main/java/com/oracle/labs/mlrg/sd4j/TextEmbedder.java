/*
 * Copyright (c) 2023 Oracle and/or its affiliates.
 *
 * The Universal Permissive License (UPL), Version 1.0
 *
 * Subject to the condition set forth below, permission is hereby granted to any
 * person obtaining a copy of this software, associated documentation and/or data
 * (collectively the "Software"), free of charge and under any and all copyright
 * rights in the Software, and any and all patent rights owned or freely
 * licensable by each licensor hereunder covering either (i) the unmodified
 * Software as contributed to or provided by such licensor, or (ii) the Larger
 * Works (as defined below), to deal in both
 *
 * (a) the Software, and
 * (b) any piece of software and/or hardware listed in the lrgrwrks.txt file if
 * one is included with the Software (each a "Larger Work" to which the Software
 * is contributed by such licensors),
 *
 * without restriction, including without limitation the rights to copy, create
 * derivative works of, display, perform, and distribute the Software and make,
 * use, sell, offer for sale, import, export, have made, and have sold the
 * Software and the Larger Work(s), and to sublicense the foregoing rights on
 * either these or other terms.
 *
 * This license is subject to the following condition:
 * The above copyright notice and either this complete permission notice or at
 * a minimum a reference to the UPL must be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.oracle.labs.mlrg.sd4j;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import java.io.IOException;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.util.Map;
import java.util.logging.Logger;

/**
 * The text embedding model, usually a CLIP variant, loaded in via ONNX Runtime.
 */
public final class TextEmbedder implements AutoCloseable {

    private static final Logger logger = Logger.getLogger(TextEmbedder.class.getName());

    /**
     * Pad token id in the second embedder in an SDXL model.
     */
    public static final int PAD_XL_TOKEN = 0;
    /**
     * Output dimensionality for Stable Diffusion v1.5 style models.
     */
    public static final int SD_1_5_DIM_SIZE = 768;
    /**
     * Output dimensionality for Stable Diffusion v2 style models.
     */
    public static final int SD_2_DIM_SIZE = 1024;
    /**
     * Output dimensionality for Stable Diffusion XL's second text encoder.
     */
    public static final int SDXL_DIM_SIZE = 1280;

    private final OrtEnvironment env;

    private final CLIPTokenizer tokenizer;

    private final OrtSession.SessionOptions textEmbedderOpts;
    private final OrtSession textEmbedder;

    private final int dimSize;

    private final boolean isXL;
    private final String tokenName;
    private final String poolName;
    private final int padToken;
    private final boolean longIds;

    /**
     * Constructs a TextEmbedder from the supplied model and tokenizer using the default session options.
     * @param tokenizerPath The path to the tokenizer directory containing {@code merges.txt} and {@code vocab.json}.
     * @param embedderPath The path to the text embedding model, usually a CLIP variant.
     * @param defaultSize The default size of the text embedding if it cannot be extracted from the model file.
     * @throws OrtException If the model could not be loaded.
     */
    public TextEmbedder(Path tokenizerPath, Path embedderPath, int defaultSize) throws OrtException {
        this(tokenizerPath, embedderPath, new OrtSession.SessionOptions(), defaultSize, false);
    }

    /**
     * Constructs a TextEmbedder from the supplied model and tokenizer.
     * <p>
     * The model is constructed using the supplied session options.
     * @param tokenizerPath The path to the tokenizer directory containing {@code merges.txt} and {@code vocab.json}.
     * @param embedderPath The path to the text embedding model, usually a CLIP variant.
     * @param embedderOpts The session options for the text embedding model.
     * @param defaultSize The default size of the text embedding if it cannot be extracted from the model file.
     * @throws OrtException If the model could not be loaded.
     * @throws IllegalArgumentException If the tokenizer could not be loaded.
     */
    public TextEmbedder(Path tokenizerPath, Path embedderPath, OrtSession.SessionOptions embedderOpts, int defaultSize, boolean isXL) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        try {
            this.tokenizer = CLIPTokenizer.fromPath(tokenizerPath);
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to load tokenizer from " + tokenizerPath, e);
        }
        this.textEmbedderOpts = embedderOpts;
        this.textEmbedder = env.createSession(embedderPath.toString(), textEmbedderOpts);
        var inputInfo = textEmbedder.getInputInfo();
        this.longIds = ((TensorInfo) inputInfo.get("input_ids").getInfo()).type == OnnxJavaType.INT64;
        var outputInfo = textEmbedder.getOutputInfo();
        this.isXL = isXL;
        if (isXL) {
            // Second model in SDXL uses penultimate layer by default, we've not implemented clip skip.
            tokenName = "hidden_states.31";
            poolName = "text_embeds";
            padToken = PAD_XL_TOKEN;
        } else {
            tokenName = "last_hidden_state";
            poolName = "pooler_output";
            padToken = CLIPTokenizer.PAD_TOKEN;
        }
        if (!outputInfo.containsKey(tokenName)) {
            throw new IllegalArgumentException("Failed to find ONNX output '" + tokenName + "' used as the token embedding in model loaded from " + embedderPath);
        }
        if (!outputInfo.containsKey(poolName)) {
            throw new IllegalArgumentException("Failed to find ONNX output '" + poolName + "' used as the pooled sentence embedding in model loaded from " + embedderPath);
        }
        int tmpSize = (int) ((TensorInfo) outputInfo.get(tokenName).getInfo()).getShape()[2];
        if (tmpSize == -1) {
            this.dimSize = defaultSize;
        } else {
            this.dimSize = tmpSize;
        }
    }

    /**
     * Returns the dimension of the token embedding.
     * @return The token embedding dimension.
     */
    public int getDimSize() {
        return dimSize;
    }

    /**
     * Returns if this is the second embedding in an SDXL model.
     * @return True if it is the second embedding.
     */
    public boolean isXL() {
        return isXL;
    }

    /**
     * Generates an int buffer containing {@link CLIPTokenizer#BOS_TOKEN}, {@link CLIPTokenizer#PAD_TOKEN} then {@link CLIPTokenizer#MAX_LENGTH} - 1 of the
     * correct pad token for the model (either {@link CLIPTokenizer#PAD_TOKEN} or {@link #PAD_XL_TOKEN}).
     * @return The unconditional tokens.
     */
    private IntBuffer unconditionalTokens() {
        IntBuffer output = IntBuffer.allocate(CLIPTokenizer.MAX_LENGTH);
        output.put(CLIPTokenizer.BOS_TOKEN);
        output.put(CLIPTokenizer.PAD_TOKEN); // EOS token for both models
        for (int pos = 2; pos < CLIPTokenizer.MAX_LENGTH; pos++) {
            output.put(padToken);
        }
        output.rewind();
        return output;
    }

    /**
     * Embeds a batch of text tokens using the embedding model.
     * @param tokenIds The text tokens.
     * @return The embedding tensor.
     * @throws OrtException If the model call failed.
     */
    private EmbeddingOutput embedTokens(IntTensor tokenIds) throws OrtException {
        Tensor<?> ids = longIds ? tokenIds.convertToLongTensor() : tokenIds;
        try (OnnxTensor input = ids.wrapForORT(env);
            OrtSession.Result output = textEmbedder.run(Map.of("input_ids", input))) {
            var tokenBuffer = ((OnnxTensor) output.get(tokenName).get()).getFloatBuffer();
            FloatTensor tokenEmbeddings = new FloatTensor(tokenBuffer, new long[]{tokenIds.shape[0], CLIPTokenizer.MAX_LENGTH, dimSize});
            var poolBuffer = ((OnnxTensor) output.get(poolName).get()).getFloatBuffer();
            FloatTensor poolEmbeddings = new FloatTensor(poolBuffer, new long[]{tokenIds.shape[0], dimSize});
            return new EmbeddingOutput(tokenEmbeddings, poolEmbeddings);
        }
    }

    /**
     * Generates an embedding of the text.
     * @param text The text to embed.
     * @param batchSize The batch size of images to generate.
     * @return A tensor of size [batchSize, 77, dimSize] and one of size [batchSize, dimSize].
     * @throws OrtException If the model call failed.
     */
    public EmbeddingOutput embedText(String text, int batchSize) throws OrtException {
        IntBuffer ids = tokenizer.tokenize(text);
        return embedText(batchSize, ids);
    }

    /**
     * Generates an embedding of both the text and the unconditional output (i.e. an empty sentence).
     * @param text The text to embed.
     * @param batchSize The batch size of images to generate.
     * @return A tensor of size [batchSize*2, 77, dimSize] and one of size [batchSize*2, dimSize].
     * @throws OrtException If the model call failed.
     */
    public EmbeddingOutput embedTextAndUncond(String text, int batchSize) throws OrtException {
        IntBuffer ids = tokenizer.tokenize(text);
        IntBuffer uncond = unconditionalTokens();
        return embedText(batchSize, ids, uncond);
    }

    /**
     * Generates an embedding of both the text and the negative text.
     * @param text The text to embed.
     * @param negative The negative text to embed.
     * @param batchSize The batch size of images to generate.
     * @return A tensor of size [batchSize*2, 77, dimSize] and one of size [batchSize*2, dimSize].
     * @throws OrtException If the model call failed.
     */
    public EmbeddingOutput embedTextAndNegative(String text, String negative, int batchSize) throws OrtException {
        IntBuffer ids = tokenizer.tokenize(text);
        IntBuffer negativeIds = tokenizer.tokenize(negative);
        return embedText(batchSize, ids, negativeIds);
    }

    /**
     * Embeds the supplied tokens.
     * @param batchSize The batch size of images to generate.
     * @param positiveTokens The positive tokens.
     * @return A tensor of size [batch_size*2, 77, dimSize].
     * @throws OrtException If the model call failed.
     */
    private EmbeddingOutput embedText(int batchSize, IntBuffer positiveTokens) throws OrtException {
        IntTensor idTensor = new IntTensor(new long[]{batchSize, CLIPTokenizer.MAX_LENGTH});
        for (int i = 0; i < batchSize; i++) {
            idTensor.buffer.put(positiveTokens);
            positiveTokens.rewind();
        }
        idTensor.buffer.rewind();
        return embedTokens(idTensor);
    }

    /**
     * Embeds the supplied tokens.
     * @param batchSize The batch size of images to generate.
     * @param positiveTokens The positive tokens.
     * @param negativeTokens The negative tokens.
     * @return A tensor of size [batch_size*2, 77, dimSize].
     * @throws OrtException If the model call failed.
     */
    private EmbeddingOutput embedText(int batchSize, IntBuffer positiveTokens, IntBuffer negativeTokens) throws OrtException {
        IntTensor idTensor = new IntTensor(new long[]{batchSize*2L, CLIPTokenizer.MAX_LENGTH});
        for (int i = 0; i < batchSize; i++) {
            idTensor.buffer.put(negativeTokens);
            negativeTokens.rewind();
        }
        for (int i = 0; i < batchSize; i++) {
            idTensor.buffer.put(positiveTokens);
            positiveTokens.rewind();
        }
        idTensor.buffer.rewind();
        return embedTokens(idTensor);
    }

    @Override
    public void close() throws OrtException {
        textEmbedder.close();
        textEmbedderOpts.close();
    }

    /**
     * Tuple for the output of the text embedding.
     * @param tokenEmbedding The token level embedding of size [batchSize, tokenMaxLength, embeddingDim].
     * @param pooledEmbedding The pooled sentence embedding of size [batchSize, embeddingDim].
     */
    public record EmbeddingOutput(FloatTensor tokenEmbedding, FloatTensor pooledEmbedding) {}
}
