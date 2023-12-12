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

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.IntBuffer;
import java.nio.file.Path;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;

/**
 * The text embedding model, usually a CLIP variant, loaded in via ONNX Runtime.
 */
public final class TextEmbedder implements AutoCloseable {

    private static final Logger logger = Logger.getLogger(TextEmbedder.class.getName());

    /**
     * Max length of the CLIP token output.
     */
    public static final int MAX_LENGTH = 77;
    /**
     * Pad and EOS token id.
     */
    public static final int PAD_TOKEN = 49407;
    /**
     * BOS token id.
     */
    public static final int BOS_TOKEN = 49406;
    /**
     * Output dimensionality.
     */
    public static final int DIM_SIZE = 768;

    /**
     * Pattern which matches linefeeds as they crash the tokenizer.
     */
    private static final Pattern NEWLINE_PATTERN = Pattern.compile("\\R");

    private final OrtEnvironment env;

    private final OrtSession.SessionOptions tokenizerOpts;
    private final OrtSession tokenizer;

    private final OrtSession.SessionOptions textEmbedderOpts;
    private final OrtSession textEmbedder;

    /**
     * Constructs a TextEmbedder from the supplied model and tokenizer using the default session options.
     * @param tokenizerPath The path to the tokenizer model.
     * @param embedderPath The path to the text embedding model, usually a CLIP variant.
     * @throws OrtException If the model could not be loaded.
     */
    public TextEmbedder(Path tokenizerPath, Path embedderPath) throws OrtException {
        this(tokenizerPath, embedderPath, new OrtSession.SessionOptions());

    }

    /**
     * Constructs a TextEmbedder from the supplied model and tokenizer.
     * <p>
     * The model is constructed using the supplied session options, the tokenizer uses the default options.
     * @param tokenizerPath The path to the tokenizer model.
     * @param embedderPath The path to the text embedding model, usually a CLIP variant.
     * @param embedderOpts The session options for the text embedding model.
     * @throws OrtException If the model could not be loaded.
     */
    public TextEmbedder(Path tokenizerPath, Path embedderPath, OrtSession.SessionOptions embedderOpts) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.tokenizerOpts = new OrtSession.SessionOptions();
        this.tokenizerOpts.registerCustomOpLibrary(System.mapLibraryName("ortextensions"));
        this.tokenizer = env.createSession(tokenizerPath.toString(), tokenizerOpts);
        this.textEmbedderOpts = embedderOpts;
        this.textEmbedder = env.createSession(embedderPath.toString(), textEmbedderOpts);
    }

    /**
     * Tokenizes the supplied text using the tokenization model.
     * @param text The text to tokenize.
     * @return An int buffer containing the token ids.
     * @throws OrtException If the tokenization model failed.
     */
    private IntBuffer tokenizeText(String text) throws OrtException {
        String textInput = NEWLINE_PATTERN.matcher(text).replaceAll("");
        try (OnnxTensor input = OnnxTensor.createTensor(env, new String[]{textInput}, new long[]{1});
             OrtSession.Result output = tokenizer.run(Map.of("string_input", input))) {
            var outputTensor = (OnnxTensor) output.get(0);
            var idBuffer = outputTensor.getLongBuffer();
            if (logger.isLoggable(Level.FINE)) {
                var sb = new StringBuilder();
                sb.append("Ids - [");
                for (int i = 0; i < idBuffer.limit(); i++) {
                    sb.append(" ");
                    sb.append(idBuffer.get());
                }
                sb.replace(sb.length() - 1, sb.length(), "]");
                logger.fine(sb.toString());
                idBuffer.rewind();
            }
            IntBuffer intBuffer = IntBuffer.allocate(MAX_LENGTH);
            int pos = 0;
            for (int i = 0; i < idBuffer.limit() && pos < intBuffer.capacity(); i++, pos++) {
                intBuffer.put((int)idBuffer.get(i));
            }
            for (; pos < intBuffer.capacity(); pos++) {
                intBuffer.put(PAD_TOKEN);
            }
            intBuffer.rewind();
            return intBuffer;
        }
    }

    /**
     * Generates an int buffer containing {@link #BOS_TOKEN} then {@link #MAX_LENGTH} - 1 {@link #PAD_TOKEN}s.
     * @return The unconditional tokens.
     */
    private IntBuffer unconditionalTokens() {
        IntBuffer output = IntBuffer.allocate(MAX_LENGTH);
        output.put(BOS_TOKEN);
        for (int pos = 1; pos < MAX_LENGTH; pos++) {
            output.put(PAD_TOKEN);
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
    private FloatTensor embedTokens(IntTensor tokenIds) throws OrtException {
        try (OnnxTensor input = OnnxTensor.createTensor(env, tokenIds.buffer(), tokenIds.shape());
            OrtSession.Result output = textEmbedder.run(Map.of("input_ids", input))) {
            var fb = ((OnnxTensor) output.get(0)).getFloatBuffer();
            return new FloatTensor(fb, new long[]{tokenIds.shape[0], MAX_LENGTH, DIM_SIZE});
        }
    }

    /**
     * Generates an embedding of both the text and the unconditional output (i.e. an empty sentence).
     * @param text The text to embed.
     * @param batchSize The batch size of images to generate.
     * @return A tensor of size [batch_size*2, 77, 768].
     * @throws OrtException If the model call failed.
     */
    public FloatTensor embedTextAndUncond(String text, int batchSize) throws OrtException {
        IntBuffer ids = tokenizeText(text);
        IntBuffer uncond = unconditionalTokens();
        return embedText(batchSize, ids, uncond);
    }

    /**
     * Generates an embedding of both the text and the negative text.
     * @param text The text to embed.
     * @param negative The negative text to embed.
     * @param batchSize The batch size of images to generate.
     * @return A tensor of size [batch_size*2, 77, 768].
     * @throws OrtException If the model call failed.
     */
    public FloatTensor embedTextAndNegative(String text, String negative, int batchSize) throws OrtException {
        IntBuffer ids = tokenizeText(text);
        IntBuffer negativeIds = tokenizeText(negative);
        return embedText(batchSize, ids, negativeIds);
    }

    /**
     * Embeds the supplied tokens.
     * @param batchSize The batch size of images to generate.
     * @param positiveTokens The positive tokens.
     * @param negativeTokens The negative tokens.
     * @return A tensor of size [batch_size*2, 77, 768].
     * @throws OrtException If the model call failed.
     */
    private FloatTensor embedText(int batchSize, IntBuffer positiveTokens, IntBuffer negativeTokens) throws OrtException {
        IntTensor idTensor = new IntTensor(new long[]{batchSize*2L, MAX_LENGTH});
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

        tokenizer.close();
        tokenizerOpts.close();
    }
}
