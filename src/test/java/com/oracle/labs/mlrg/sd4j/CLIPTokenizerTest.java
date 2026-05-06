/*
 * Copyright (c) 2026 Oracle and/or its affiliates.
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
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CLIPTokenizerTest {

    private static final int HELLO = 100;
    private static final int WORLD = 101;
    private static final int HI = 102;
    private static final int EXCLAMATION = 103;
    private static final int A = 104;
    private static final int AB = 106;

    @TempDir
    Path tempDir;

    @Test
    void tokenizesLowercaseBpeAndPunctuation() throws IOException {
        CLIPTokenizer tokenizer = tokenizer(tempDir);

        IntBuffer tokens = tokenizer.tokenize("HI!");

        assertFixedLength(tokens);
        assertEquals(CLIPTokenizer.BOS_TOKEN, tokens.get(0));
        assertEquals(HI, tokens.get(1));
        assertEquals(EXCLAMATION, tokens.get(2));
        assertPaddingFrom(tokens, 3);
    }

    @Test
    void tokenizesMultipleWordsWithBpeMerges() throws IOException {
        CLIPTokenizer tokenizer = tokenizer(tempDir);

        IntBuffer tokens = tokenizer.tokenize("Hello world");

        assertFixedLength(tokens);
        assertEquals(CLIPTokenizer.BOS_TOKEN, tokens.get(0));
        assertEquals(HELLO, tokens.get(1));
        assertEquals(WORLD, tokens.get(2));
        assertPaddingFrom(tokens, 3);
    }

    @Test
    void removesNewlinesBeforeTokenizing() throws IOException {
        CLIPTokenizer tokenizer = tokenizer(tempDir);

        IntBuffer tokens = tokenizer.tokenize("a\nb");

        assertFixedLength(tokens);
        assertEquals(CLIPTokenizer.BOS_TOKEN, tokens.get(0));
        assertEquals(AB, tokens.get(1));
        assertPaddingFrom(tokens, 2);
    }

    @Test
    void emptyInputReturnsBosAndPadding() throws IOException {
        CLIPTokenizer tokenizer = tokenizer(tempDir);

        IntBuffer tokens = tokenizer.tokenize("");

        assertFixedLength(tokens);
        assertEquals(CLIPTokenizer.BOS_TOKEN, tokens.get(0));
        assertPaddingFrom(tokens, 1);
    }

    @Test
    void truncatesLongInputToMaxLength() throws IOException {
        CLIPTokenizer tokenizer = tokenizer(tempDir);
        String input = IntStream.range(0, 100).mapToObj(i -> "a").collect(Collectors.joining(" "));

        IntBuffer tokens = tokenizer.tokenize(input);

        assertFixedLength(tokens);
        assertEquals(CLIPTokenizer.BOS_TOKEN, tokens.get(0));
        for (int i = 1; i < CLIPTokenizer.MAX_LENGTH; i++) {
            assertEquals(A, tokens.get(i));
        }
    }

    @Test
    void validatesSpecialTokenIds() throws IOException {
        writeMerges(tempDir);
        Files.writeString(tempDir.resolve("vocab.json"), """
                {
                  "<|startoftext|>": 1,
                  "<|endoftext|>": 49407
                }
                """);

        assertThrows(IllegalArgumentException.class, () -> CLIPTokenizer.fromPath(tempDir));
    }

    @Disabled
    @Test
    void matchesOnnxTokenizerOutput() throws IOException, OrtException {
        Path tokenizerPath = Path.of("sdxl/tokenizer");
        Path onnxTokenizerPath = Path.of("text_tokenizer/custom_op_cliptok.onnx");
        CLIPTokenizer tokenizer = CLIPTokenizer.fromPath(tokenizerPath);

        String[] sentences = {
                "",
                "a photo of an astronaut riding a horse",
                "The quick brown fox jumps over the lazy dog.",
                "HI! This shouldn't break contractions, punctuation, or CAPS.",
                "line one\nline two",
                "unicode accents: caf\u00E9 na\u00EFve fa\u00E7ade",
                "symbols <>[]{} -- 12345 67.89",
                IntStream.range(0, 120).mapToObj(i -> "long").collect(Collectors.joining(" "))
        };

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.registerCustomOpLibrary("./" + System.mapLibraryName("ortextensions"));
            try (OrtSession session = env.createSession(onnxTokenizerPath.toString(), options)) {
                for (String sentence : sentences) {
                    assertArrayEquals(
                            onnxTokens(env, session, sentence),
                            tokens(tokenizer.tokenize(sentence)),
                            () -> "Token mismatch for sentence: " + sentence);
                }
            }
        }
    }

    private static CLIPTokenizer tokenizer(Path tokenizerPath) throws IOException {
        writeMerges(tokenizerPath);
        Files.writeString(tokenizerPath.resolve("vocab.json"), """
                {
                  "<|startoftext|>": 49406,
                  "<|endoftext|>": 49407,
                  "hello</w>": 100,
                  "world</w>": 101,
                  "hi</w>": 102,
                  "!</w>": 103,
                  "a</w>": 104,
                  "b</w>": 105,
                  "ab</w>": 106
                }
                """);
        return CLIPTokenizer.fromPath(tokenizerPath);
    }

    private static int[] onnxTokens(OrtEnvironment env, OrtSession session, String sentence) throws OrtException {
        String inputText = sentence.replaceAll("\\R", "");
        try (OnnxTensor input = OnnxTensor.createTensor(env, new String[]{inputText}, new long[]{1});
             OrtSession.Result output = session.run(Map.of("string_input", input))) {
            LongBuffer ids = ((OnnxTensor) output.get(0)).getLongBuffer();
            int[] tokens = new int[CLIPTokenizer.MAX_LENGTH];
            int pos = 0;
            for (int i = 0; i < ids.limit() && pos < tokens.length; i++, pos++) {
                tokens[pos] = (int) ids.get(i);
            }
            for (; pos < tokens.length; pos++) {
                tokens[pos] = CLIPTokenizer.PAD_TOKEN;
            }
            return tokens;
        }
    }

    private static int[] tokens(IntBuffer buffer) {
        int[] output = new int[buffer.limit()];
        for (int i = 0; i < output.length; i++) {
            output[i] = buffer.get(i);
        }
        return output;
    }

    private static void writeMerges(Path tokenizerPath) throws IOException {
        Files.writeString(tokenizerPath.resolve("merges.txt"), """
                #version: 0.2
                h i</w>
                h e
                he l
                hel l
                hell o</w>
                w o
                wo r
                wor l
                worl d</w>
                a b</w>
                """);
    }

    private static void assertFixedLength(IntBuffer tokens) {
        assertEquals(0, tokens.position());
        assertEquals(CLIPTokenizer.MAX_LENGTH, tokens.limit());
        assertEquals(CLIPTokenizer.MAX_LENGTH, tokens.capacity());
    }

    private static void assertPaddingFrom(IntBuffer tokens, int start) {
        for (int i = start; i < CLIPTokenizer.MAX_LENGTH; i++) {
            assertEquals(CLIPTokenizer.PAD_TOKEN, tokens.get(i));
        }
    }
}
