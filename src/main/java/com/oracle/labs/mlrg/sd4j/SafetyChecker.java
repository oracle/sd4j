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

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * The safety checker which tags images which are unsuitable for work.
 */
public final class SafetyChecker implements AutoCloseable {
    private static final Logger logger = Logger.getLogger(SafetyChecker.class.getName());

    /**
     * Possible outputs from the Safety Checker
     */
    public enum CheckerOutput {
        /**
         * Image is predicted to be safe for work.
         */
        SFW,
        /**
         * Image is predicted to be not safe for work.
         */
        NSFW
    }

    private final OrtEnvironment env;

    private final OrtSession.SessionOptions resizerOpts;
    private final OrtSession resizer;

    private final OrtSession.SessionOptions checkerOpts;
    private final OrtSession checker;

    /**
     * This is the output of {@code com.oracle.labs.mlrg.sd4j.ResizeModelGenerator}, stored as a String for easier packaging.
     */
    public static final String resizerModel =
            "CAgiHGNvbS5odWdnaW5nZmFjZS50cmFuc2Zvcm1lcnMoADIUQ0xJUCBJbWFnZSBQcm9jZXNzb3I6zQcKMgoFaW5wdXQKCWNvbnN0LXR3bxIPZGl2aWRlLTAtb3V0cHV0GghkaXZpZGUtMCIDRGl2CjcKD2RpdmlkZS0wLW91dHB1dAoKY29uc3QtaGFsZhIMYWRkLTAtb3V0cHV0GgVhZGQtMCIDQWRkCjQKDGFkZC0wLW91dHB1dAoKY29uc3QtemVybxIMbWF4LTAtb3V0cHV0GgVtYXgtMCIDTWF4CjMKDG1heC0wLW91dHB1dAoJY29uc3Qtb25lEgxtaW4tMC1vdXRwdXQaBW1pbi0wIgNNaW4KJwoFaW5wdXQSDnNoYXBlLTAtb3V0cHV0GgdzaGFwZS0wIgVTaGFwZQpGCg5zaGFwZS0wLW91dHB1dAoRY29uc3QtdGVuc29yLXplcm8SD2dhdGhlci0wLW91dHB1dBoIZ2F0aGVyLTAiBkdhdGhlcgpNCg9nYXRoZXItMC1vdXRwdXQKCmltYWdlLXNpemUSD2NvbmNhdC0wLW91dHB1dBoIY29uY2F0LTAiBkNvbmNhdCoLCgRheGlzGACgAQIKWQoMbWluLTAtb3V0cHV0CgAKAAoPY29uY2F0LTAtb3V0cHV0Eg9yZXNpemUtMC1vdXRwdXQaCHJlc2l6ZS0wIgZSZXNpemUqEQoEbW9kZSIGbGluZWFyoAEDCkQKD3Jlc2l6ZS0wLW91dHB1dAoNY2hhbm5lbC1tZWFucxIRc3VidHJhY3QtMC1vdXRwdXQaCnN1YnRyYWN0LTAiA1N1Ygo7ChFzdWJ0cmFjdC0wLW91dHB1dAoPY2hhbm5lbC1zdGRkZXZzEgZvdXRwdXQaCGRpdmlkZS0xIgNEaXYSEkNMSVBJbWFnZVByb2Nlc3NvcioXCAMQBzoFA+AB4AFCCmltYWdlLXNpemUqExABIgQAAABAQgljb25zdC10d28qFBABIgQAAAA/Qgpjb25zdC1oYWxmKhMQASIEAACAP0IJY29uc3Qtb25lKhQQASIEAAAAAEIKY29uc3QtemVybyoaCAEQBzoBAEIRY29uc3QtdGVuc29yLXplcm8qJwgBCAMIAQgBEAEiDDqB9j5eaOo+/wDRPkINY2hhbm5lbC1tZWFucyopCAEIAwgBCAEQASIM0ImJPnTJhT6oMo0+Qg9jaGFubmVsLXN0ZGRldnNaNAoFaW5wdXQSKwopCAESJQoMEgpiYXRjaF9zaXplCgIIAwoIEgZoZWlnaHQKBxIFd2lkdGhiLAoGb3V0cHV0EiIKIAgBEhwKDBIKYmF0Y2hfc2l6ZQoCCAMKAwjgAQoDCOABQgIQEg==";

    /**
     * Constructs a safety checker from the supplied model path.
     * @param checkerModelPath The model path.
     * @throws OrtException If the model failed to load.
     */
    public SafetyChecker(Path checkerModelPath) throws OrtException {
        this(checkerModelPath, new OrtSession.SessionOptions());
    }

    /**
     * Constructs a safety checker from the supplied model path, using the supplied options (e.g., to enable CUDA).
     *
     * @param checkerModelPath The model path.
     * @param opts The session options.
     * @throws OrtException If the model failed to load.
     */
    public SafetyChecker(Path checkerModelPath, OrtSession.SessionOptions opts) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.resizerOpts = new OrtSession.SessionOptions();
        this.resizer = env.createSession(Base64.getDecoder().decode(resizerModel), resizerOpts);
        this.checkerOpts = opts;
        this.checker = env.createSession(checkerModelPath.toString(), checkerOpts);
    }

    /**
     * Returns SFW/NSFW depending on the prediction of the safety classifier.
     * @param decodedImage The image as a float.
     * @return A list with enums one per image in the batch.
     * @throws OrtException If the model run failed.
     */
    public List<CheckerOutput> check(FloatTensor decodedImage) throws OrtException {
        logger.info("Running safety check");
        try (var inputTensor = OnnxTensor.createTensor(env, decodedImage.buffer, decodedImage.shape);
             var resizedResult = resizer.run(Map.of("input", inputTensor))) {
            // Need to split the result into different tensors, one per image
            // The safety checker doesn't like batches for some reason
            var output = new ArrayList<CheckerOutput>();
            var resizedTensor = (OnnxTensor) resizedResult.get(0);
            var javaSideTensor = new FloatTensor(resizedTensor.getFloatBuffer(), resizedTensor.getInfo().getShape());
            var splitResizedTensor = javaSideTensor.split(new long[]{1, 3, 224, 224});
            for (FloatTensor f : splitResizedTensor) {
                try (var result = checker.run(Map.of("clip_input", f.wrapForORT(env), "images", inputTensor))) {
                    var outputTensor = (OnnxTensor) result.get(1);
                    var bools = (boolean[]) outputTensor.getValue();
                    if (bools.length != 1) {
                        throw new IllegalStateException("Expected a single prediction, found " + bools.length);
                    }
                    output.add(bools[0] ? CheckerOutput.NSFW : CheckerOutput.SFW);
                }
            }
            return output;
        }
    }

    @Override
    public void close() throws OrtException {
        resizer.close();
        resizerOpts.close();
        checker.close();
        checkerOpts.close();
    }
}
