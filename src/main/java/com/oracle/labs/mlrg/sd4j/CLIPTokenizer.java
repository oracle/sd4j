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

import java.io.IOException;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A CLIP byte-level Byte-Pair Encoding tokenizer.
 */
public final class CLIPTokenizer {

    /**
     * Max length of the CLIP token output.
     */
    public static final int MAX_LENGTH = 77;
    /**
     * Pad and EOS token id in regular SD models.
     */
    public static final int PAD_TOKEN = 49407;
    /**
     * BOS token id.
     */
    public static final int BOS_TOKEN = 49406;

    private static final String START_OF_TEXT = "<|startoftext|>";
    private static final String END_OF_TEXT = "<|endoftext|>";
    private static final String END_OF_WORD = "</w>";

    private static final Pattern NEWLINE_PATTERN = Pattern.compile("\\R");
    private static final Pattern TOKEN_PATTERN = Pattern.compile(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]+|[^\\s\\p{L}\\p{N}]+",
            Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE | Pattern.UNICODE_CHARACTER_CLASS);

    private static final String[] BYTE_ENCODER = bytesToUnicode();

    private final Map<String, Integer> vocab;
    private final Map<Merge, Integer> ranks;
    private final Map<String, String> bpeCache = new ConcurrentHashMap<>();

    private CLIPTokenizer(List<Merge> merges, Map<String, Integer> vocab) {
        Objects.requireNonNull(merges, "merges");
        Objects.requireNonNull(vocab, "vocab");
        this.vocab = Map.copyOf(vocab);
        this.ranks = new HashMap<>();
        for (int i = 0; i < merges.size(); i++) {
            this.ranks.put(merges.get(i), i);
        }
        validateSpecialToken(START_OF_TEXT, BOS_TOKEN);
        validateSpecialToken(END_OF_TEXT, PAD_TOKEN);
    }

    /**
     * Loads a CLIP tokenizer from a Hugging Face tokenizer directory.
     * @param tokenizerPath The directory containing {@code merges.txt} and {@code vocab.json}.
     * @return A CLIP tokenizer.
     * @throws IOException If the tokenizer files could not be read or parsed.
     */
    public static CLIPTokenizer fromPath(Path tokenizerPath) throws IOException {
        List<Merge> merges = parseMerges(tokenizerPath.resolve("merges.txt"));
        Map<String, Integer> vocab = parseVocab(Files.readString(tokenizerPath.resolve("vocab.json"), StandardCharsets.UTF_8));
        return new CLIPTokenizer(merges, vocab);
    }

    /**
     * Tokenizes the supplied input into a fixed length CLIP token buffer.
     * @param input The input text.
     * @return A rewound int buffer containing exactly {@link CLIPTokenizer#MAX_LENGTH} tokens.
     */
    public IntBuffer tokenize(String input) {
        Objects.requireNonNull(input, "input");
        String text = NEWLINE_PATTERN.matcher(input).replaceAll("").toLowerCase(Locale.ROOT);
        List<Integer> tokenIds = new ArrayList<>(MAX_LENGTH);
        tokenIds.add(BOS_TOKEN);

        Matcher matcher = TOKEN_PATTERN.matcher(text);
        while (matcher.find() && tokenIds.size() < MAX_LENGTH) {
            String encodedToken = encodeToken(matcher.group());
            String bpeToken = bpeCache.computeIfAbsent(encodedToken, this::bytePairEncode);
            for (String token : bpeToken.split(" ")) {
                if (tokenIds.size() == MAX_LENGTH) {
                    break;
                }
                tokenIds.add(vocabId(token));
            }
        }

        if (tokenIds.size() < MAX_LENGTH) {
            tokenIds.add(PAD_TOKEN);
        }
        IntBuffer output = IntBuffer.allocate(MAX_LENGTH);
        for (int tokenId : tokenIds) {
            output.put(tokenId);
        }
        while (output.hasRemaining()) {
            output.put(PAD_TOKEN);
        }
        output.rewind();
        return output;
    }

    private static List<Merge> parseMerges(Path mergesPath) throws IOException {
        List<String> lines = Files.readAllLines(mergesPath, StandardCharsets.UTF_8);
        List<Merge> merges = new ArrayList<>(Math.max(0, lines.size() - 1));
        for (String line : lines) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            int split = line.indexOf(' ');
            if (split < 1 || split == line.length() - 1) {
                throw new IOException("Malformed merge line in " + mergesPath + ": " + line);
            }
            merges.add(new Merge(line.substring(0, split), line.substring(split + 1)));
        }
        return merges;
    }

    private static Map<String, Integer> parseVocab(String json) throws IOException {
        return new VocabParser(json).parse();
    }

    private static String[] bytesToUnicode() {
        boolean[] original = new boolean[256];
        List<Integer> bytes = new ArrayList<>(256);
        addByteRange(bytes, original, '!', '~');
        addByteRange(bytes, original, '¡', '¬');
        addByteRange(bytes, original, '®', 'ÿ');

        List<Integer> codePoints = new ArrayList<>(bytes);
        int next = 0;
        for (int b = 0; b < 256; b++) {
            if (!original[b]) {
                bytes.add(b);
                codePoints.add(256 + next);
                next++;
            }
        }

        String[] encoder = new String[256];
        for (int i = 0; i < bytes.size(); i++) {
            encoder[bytes.get(i)] = new String(Character.toChars(codePoints.get(i)));
        }
        return encoder;
    }

    private static void addByteRange(List<Integer> bytes, boolean[] original, int start, int end) {
        for (int b = start; b <= end; b++) {
            bytes.add(b);
            original[b] = true;
        }
    }

    private static Set<Merge> getPairs(List<String> word) {
        Set<Merge> pairs = new HashSet<>();
        for (int i = 0; i < word.size() - 1; i++) {
            pairs.add(new Merge(word.get(i), word.get(i + 1)));
        }
        return pairs;
    }

    private String encodeToken(String token) {
        byte[] utf8 = token.getBytes(StandardCharsets.UTF_8);
        StringBuilder output = new StringBuilder(utf8.length);
        for (byte b : utf8) {
            output.append(BYTE_ENCODER[b & 0xFF]);
        }
        return output.toString();
    }

    private String bytePairEncode(String token) {
        List<String> word = new ArrayList<>(token.length());
        for (int i = 0; i < token.length() - 1; i++) {
            word.add(String.valueOf(token.charAt(i)));
        }
        word.add(token.charAt(token.length() - 1) + END_OF_WORD);

        while (word.size() > 1) {
            Merge best = null;
            int bestRank = Integer.MAX_VALUE;
            for (Merge pair : getPairs(word)) {
                Integer rank = ranks.get(pair);
                if (rank != null && rank < bestRank) {
                    best = pair;
                    bestRank = rank;
                }
            }
            if (best == null) {
                break;
            }
            word = mergePair(word, best);
        }
        return String.join(" ", word);
    }

    private static List<String> mergePair(List<String> word, Merge merge) {
        List<String> merged = new ArrayList<>(word.size());
        int i = 0;
        while (i < word.size()) {
            if (i < word.size() - 1 && word.get(i).equals(merge.first()) && word.get(i + 1).equals(merge.second())) {
                merged.add(merge.first() + merge.second());
                i += 2;
            } else {
                merged.add(word.get(i));
                i++;
            }
        }
        return merged;
    }

    private int vocabId(String token) {
        Integer id = vocab.get(token);
        if (id == null) {
            throw new IllegalArgumentException("Token '" + token + "' is not present in the tokenizer vocabulary.");
        }
        return id;
    }

    private void validateSpecialToken(String token, int expectedId) {
        Integer actual = vocab.get(token);
        if (actual == null) {
            throw new IllegalArgumentException("Tokenizer vocabulary is missing required token '" + token + "'.");
        }
        if (actual != expectedId) {
            throw new IllegalArgumentException("Tokenizer vocabulary maps '" + token + "' to " + actual + " instead of " + expectedId + ".");
        }
    }

    private record Merge(String first, String second) {}

    private static final class VocabParser {
        private final String json;
        private int pos;

        private VocabParser(String json) {
            this.json = json;
        }

        private Map<String, Integer> parse() throws IOException {
            Map<String, Integer> output = new HashMap<>();
            skipWhitespace();
            expect('{');
            skipWhitespace();
            if (peek('}')) {
                pos++;
                return output;
            }
            while (true) {
                skipWhitespace();
                String key = parseString();
                skipWhitespace();
                expect(':');
                skipWhitespace();
                int value = parseInt();
                output.put(key, value);
                skipWhitespace();
                if (peek(',')) {
                    pos++;
                    continue;
                }
                expect('}');
                break;
            }
            skipWhitespace();
            if (pos != json.length()) {
                throw error("Unexpected trailing content");
            }
            return output;
        }

        private String parseString() throws IOException {
            expect('"');
            StringBuilder output = new StringBuilder();
            while (pos < json.length()) {
                char c = json.charAt(pos++);
                if (c == '"') {
                    return output.toString();
                }
                if (c == '\\') {
                    if (pos == json.length()) {
                        throw error("Unterminated escape sequence");
                    }
                    output.append(parseEscape(json.charAt(pos++)));
                } else {
                    output.append(c);
                }
            }
            throw error("Unterminated string");
        }

        private char parseEscape(char escape) throws IOException {
            return switch (escape) {
                case '"', '\\', '/' -> escape;
                case 'b' -> '\b';
                case 'f' -> '\f';
                case 'n' -> '\n';
                case 'r' -> '\r';
                case 't' -> '\t';
                case 'u' -> parseUnicodeEscape();
                default -> throw error("Unsupported escape sequence \\" + escape);
            };
        }

        private char parseUnicodeEscape() throws IOException {
            if (pos + 4 > json.length()) {
                throw error("Incomplete unicode escape");
            }
            int value = 0;
            for (int i = 0; i < 4; i++) {
                int digit = Character.digit(json.charAt(pos++), 16);
                if (digit < 0) {
                    throw error("Invalid unicode escape");
                }
                value = (value << 4) + digit;
            }
            return (char) value;
        }

        private int parseInt() throws IOException {
            int start = pos;
            if (peek('-')) {
                pos++;
            }
            while (pos < json.length() && Character.isDigit(json.charAt(pos))) {
                pos++;
            }
            if (start == pos || (json.charAt(start) == '-' && start + 1 == pos)) {
                throw error("Expected integer");
            }
            try {
                return Integer.parseInt(json.substring(start, pos));
            } catch (NumberFormatException e) {
                throw error("Invalid integer");
            }
        }

        private void expect(char expected) throws IOException {
            if (pos == json.length() || json.charAt(pos) != expected) {
                throw error("Expected '" + expected + "'");
            }
            pos++;
        }

        private boolean peek(char expected) {
            return pos < json.length() && json.charAt(pos) == expected;
        }

        private void skipWhitespace() {
            while (pos < json.length() && Character.isWhitespace(json.charAt(pos))) {
                pos++;
            }
        }

        private IOException error(String message) {
            return new IOException(message + " at offset " + pos + ".");
        }
    }
}
