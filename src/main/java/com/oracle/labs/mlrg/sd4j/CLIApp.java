/*
 * Copyright (c) 2023, 2024, Oracle and/or its affiliates.
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

import ai.onnxruntime.OrtException;

import java.io.IOException;
import java.util.List;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * A short command line demo that makes two images, one with only a positive prompt and one with a positive and negative prompt.
 */
public final class CLIApp {
    private static final Logger logger = Logger.getLogger(CLIApp.class.getName());

    /**
     * Private constructor for no instance class.
     */
    private CLIApp() {}

    /**
     * Demo entrypoint.
     * @param args The CLI args.
     * @throws OrtException If ORT fails.
     * @throws IOException If the image files could not be written out.
     */
    public static void main(String[] args) throws OrtException, IOException {
        Optional<SD4J.SD4JConfig> config = SD4J.SD4JConfig.parseArgs(args);
        if (config.isEmpty()) {
            System.out.println(SD4J.SD4JConfig.help());
            System.exit(1);
        }

        SD4J sd = SD4J.factory(config.get());

        String text = "Professional photo of a green tree surrounded by purple flowers and a sunset in a red sky";

        int seed = 42;
        List<SD4J.SDImage> images = sd.generateImage(30, text, "", 7.5f, 1, new SD4J.ImageSize(512, 512), seed);
        String output = "output-"+seed+".png";
        logger.info("Saving to " + output);
        SD4J.save(images.get(0), output);

        String negativeText = "red tree, green sky";
        images = sd.generateImage(30, text, negativeText, 7.5f, 1, new SD4J.ImageSize(512, 512), seed);
        output = "output-neg-"+seed+".png";
        logger.info("Saving to " + output);
        SD4J.save(images.get(0), output);

        sd.close();
    }

}
