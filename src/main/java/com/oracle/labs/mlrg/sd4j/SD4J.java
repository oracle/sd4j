/*
 * Copyright (c) 2023, 2025, Oracle and/or its affiliates.
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

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.OrtCUDAProviderOptions;
import ai.onnxruntime.providers.CoreMLFlags;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageTypeSpecifier;
import javax.imageio.ImageWriter;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.metadata.IIOMetadataNode;
import javax.imageio.stream.FileImageOutputStream;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.logging.Logger;

/**
 * A stable diffusion pipeline using ONNX Runtime.
 */
public final class SD4J implements AutoCloseable {
    private static final Logger logger = Logger.getLogger(SD4J.class.getName());

    private final String modelName;
    private final TextEmbedder embedder;
    private final TextEmbedder embedderXL;
    private final UNet unet;
    private final VAEDecoder vae;
    private final SafetyChecker safety;

    /**
     * Constructs a stable diffusion pipeline from the supplied models.
     * @param modelName The model name.
     * @param embedder The text embedding model. Usually a CLIP variant.
     * @param unet The UNet model which performs the inverse diffusion.
     * @param vae The VAE model which translates from latent space to pixel space.
     * @param safety The safety checker model which checks that the generated image is SFW.
     */
    public SD4J(String modelName, TextEmbedder embedder, UNet unet, VAEDecoder vae, SafetyChecker safety) {
        this(modelName, embedder, null, unet, vae, safety);
    }

    /**
     * Constructs a stable diffusion xl pipeline from the supplied models.
     *
     * <p>Set {@code embedderXL} to null to get a standard stable diffusion pipeline.
     *
     * @param modelName The model name.
     * @param embedder The text embedding model. Usually a CLIP variant.
     * @param embedderXL The second text embedding model. Usually a CLIP variant.
     * @param unet The UNet model which performs the inverse diffusion.
     * @param vae The VAE model which translates from latent space to pixel space.
     * @param safety The safety checker model which checks that the generated image is SFW.
     */
    public SD4J(String modelName, TextEmbedder embedder, TextEmbedder embedderXL, UNet unet, VAEDecoder vae, SafetyChecker safety) {
        this.modelName = modelName;
        this.embedder = embedder;
        this.embedderXL = embedderXL;
        this.unet = unet;
        this.vae = vae;
        this.safety = safety;
    }

    /**
     * Saves the buffered image to the supplied path as a png.
     * @param image The image.
     * @param filename The filename to save to.
     * @throws IOException If the file save failed.
     */
    public static void save(SDImage image, String filename) throws IOException {
        File f = new File(filename);
        save(image, f);
    }

    /**
     * Saves the buffered image to the supplied path as a png.
     * @param image The image.
     * @param filename The path to save to.
     * @throws IOException If the file save failed.
     */
    public static void save(SDImage image, Path filename) throws IOException {
        save(image, filename.toFile());
    }

    /**
     * Saves the buffered image to the supplied path as a png.
     * @param image The image.
     * @param file The file to save to.
     * @throws IOException If the file save failed.
     */
    public static void save(SDImage image, File file) throws IOException {
        var imType = ImageTypeSpecifier.createFromRenderedImage(image.image());
        Iterator<ImageWriter> writers = ImageIO.getImageWriters(imType, "png");

        if (writers.hasNext()) {
            var writer = writers.next();
            try (var ios = new FileImageOutputStream(file)) {
                var metadata = writer.getDefaultImageMetadata(imType, writer.getDefaultWriteParam());
                writeMetadata(image, metadata);
                var ioimage = new IIOImage(image.image(), List.of(), metadata);
                writer.setOutput(ios);
                writer.write(ioimage);
            }
        } else {
            throw new IllegalStateException("No writer for 'png' found.");
        }
    }

    private static void writeMetadata(SDImage image, IIOMetadata blankMetadata) throws IOException {
        IIOMetadataNode textEntry = new IIOMetadataNode("tEXtEntry");
        textEntry.setAttribute("keyword", "parameters");
        textEntry.setAttribute("value", image.metadataDescription());

        IIOMetadataNode text = new IIOMetadataNode("tEXt");
        text.appendChild(textEntry);

        IIOMetadataNode root = new IIOMetadataNode(blankMetadata.getNativeMetadataFormatName());
        root.appendChild(text);

        blankMetadata.mergeTree(blankMetadata.getNativeMetadataFormatName(), root);
    }

    /**
     * Generates a batch of images from the supplied prompts and parameters.
     * <p>
     * Defaults to the LMS scheduler.
     * @param numInferenceSteps The number of diffusion inference steps to take (commonly 20-50 for LMS and Euler Ancestral).
     * @param text The text prompt.
     * @param negativeText The negative text prompt which the image should not contain.
     * @param guidanceScale The strength of the classifier-free guidance (i.e., how much should the image represent the text prompt).
     * @param batchSize The number of images to generate.
     * @param size The image size.
     * @param seed The RNG seed, fixing the seed should produce identical images.
     * @return A list of generated images.
     */
    public List<SDImage> generateImage(int numInferenceSteps, String text, String negativeText, float guidanceScale, int batchSize, ImageSize size, int seed) {
        return generateImage(numInferenceSteps, text, negativeText, guidanceScale, batchSize, size, seed, Schedulers.LMS, (Integer a) -> {});
    }

    /**
     * Generates a batch of images from the supplied generation request.
     * @param request The image generation request.
     * @return A list of generated images.
     */
    public List<SDImage> generateImage(Request request) {
        return generateImage(request, (Integer a) -> {});
    }

    /**
     * Generates a batch of images from the supplied prompts and parameters.
     * @param numInferenceSteps The number of diffusion inference steps to take (commonly 20-50 for LMS and Euler Ancestral).
     * @param text The text prompt.
     * @param negativeText The negative text prompt which the image should not contain.
     * @param guidanceScale The strength of the classifier-free guidance (i.e., how much should the image represent the text prompt).
     * @param batchSize The number of images to generate.
     * @param size The image size.
     * @param seed The RNG seed, fixing the seed should produce identical images.
     * @param scheduler The diffusion scheduling algorithm.
     * @param progressCallback A supplier which can be used to update a GUI. It is called after each diffusion step with the current step count.
     * @return A list of generated images.
     */
    public List<SDImage> generateImage(int numInferenceSteps, String text, String negativeText, float guidanceScale, int batchSize, ImageSize size, int seed, Schedulers scheduler, Consumer<Integer> progressCallback) {
        return generateImage(numInferenceSteps, text, negativeText, guidanceScale, batchSize, size, seed, scheduler, null, -1, progressCallback);
    }

    /**
     * Generates a batch of images from the supplied prompts and parameters.
     * @param numInferenceSteps The number of diffusion inference steps to take (commonly 20-50 for LMS and Euler Ancestral).
     * @param text The text prompt.
     * @param negativeText The negative text prompt which the image should not contain.
     * @param guidanceScale The strength of the classifier-free guidance (i.e., how much should the image represent the text prompt).
     * @param batchSize The number of images to generate.
     * @param size The image size.
     * @param seed The RNG seed, fixing the seed should produce identical images.
     * @param scheduler The diffusion scheduling algorithm.
     * @param intermediateOutputPath The output path for the intermediate images, set to null for no intermediate images.
     * @param intermediateTimesteps The number of timesteps between intermediate images, set to -1 for no intermediate images.
     * @param progressCallback A supplier which can be used to update a GUI. It is called after each diffusion step with the current step count.
     * @return A list of generated images.
     */
    public List<SDImage> generateImage(int numInferenceSteps, String text, String negativeText, float guidanceScale, int batchSize, ImageSize size, int seed, Schedulers scheduler, Path intermediateOutputPath, int intermediateTimesteps, Consumer<Integer> progressCallback) {
        var request = new Request(text, negativeText, numInferenceSteps, guidanceScale, seed, size, scheduler, batchSize, intermediateOutputPath, intermediateTimesteps);
        return generateImage(request, progressCallback);
    }

    /**
     * Generates a batch of images from the supplied generation request.
     * @param request The image generation request.
     * @param progressCallback A supplier which can be used to update a GUI. It is called after each diffusion step with the current step count.
     * @return A list of generated images.
     */
    public List<SDImage> generateImage(Request request, Consumer<Integer> progressCallback) {
        try {
            TextEmbedder.EmbeddingOutput firstEmbedding;
            TextEmbedder.EmbeddingOutput secondEmbedding = null;
            if (request.guidance() < 1.0) {
                logger.info("Generating image for '" + request.text() + "', without guidance");
                firstEmbedding = embedder.embedText(request.text(), request.batchSize());
                if (embedderXL != null) {
                    secondEmbedding = embedderXL.embedText(request.text(), request.batchSize());
                }
            } else if (request.negText().isBlank()) {
                logger.info("Generating image for '" + request.text() + "', with guidance");
                firstEmbedding = embedder.embedTextAndUncond(request.text(), request.batchSize());
                if (embedderXL != null) {
                    secondEmbedding = embedderXL.embedTextAndUncond(request.text(), request.batchSize());
                }
            } else {
                logger.info("Generating image for '" + request.text() + "', with negative text '" + request.negText() + "'");
                firstEmbedding = embedder.embedTextAndNegative(request.text(), request.negText(), request.batchSize());
                if (embedderXL != null) {
                    secondEmbedding = embedderXL.embedTextAndNegative(request.text(), request.negText(), request.batchSize());
                }
            }
            FloatTensor textEmbedding;
            FloatTensor pooledEmbedding;
            float latentScalar;
            if (embedderXL != null) {
                // SDXL
                textEmbedding = FloatTensor.concat(firstEmbedding.tokenEmbedding(), secondEmbedding.tokenEmbedding());
                pooledEmbedding = secondEmbedding.pooledEmbedding();
                latentScalar = VAEDecoder.SDXL_LATENT_SCALAR;
            } else {
                // SD
                textEmbedding = firstEmbedding.tokenEmbedding();
                pooledEmbedding = null;
                latentScalar = VAEDecoder.SD_LATENT_SCALAR;
            }
            logger.info("Generated embedding");
            BiConsumer<FloatTensor, Integer> saveCallback = createSaveCallback(request, latentScalar);
            FloatTensor latents = unet.inference(request.steps(), textEmbedding, pooledEmbedding,
                    request.guidance(), request.batchSize(), request.size().height(), request.size().width(),
                    request.seed(), progressCallback, request.scheduler(), saveCallback);
            logger.info("Generated latents");
            boolean[] isValid = new boolean[request.batchSize()];
            Arrays.fill(isValid, true);
            List<BufferedImage> image;
            if (safety != null) {
                FloatTensor decoded = vae.decoder(latents, latentScalar);
                List<SafetyChecker.CheckerOutput> checks = safety.check(decoded);
                List<BufferedImage> tmp = VAEDecoder.convertToBufferedImage(decoded);
                image = new ArrayList<>();
                for (int i = 0; i < tmp.size(); i++) {
                    logger.info("SafetyChecker says '" + checks.get(i) + "'");
                    image.add(tmp.get(i));
                    if (checks.get(i) == SafetyChecker.CheckerOutput.NSFW) {
                        isValid[i] = false;
                    }
                }
            } else {
                image = vae.decodeToBufferedImage(latents, latentScalar);
            }
            logger.info("Generated images");
            return wrap(image, request, isValid);
        } catch (OrtException e) {
            throw new IllegalStateException("Model inference failed.", e);
        }
    }

    private BiConsumer<FloatTensor, Integer> createSaveCallback(Request request, float latentScalar) {
        BiConsumer<FloatTensor, Integer> saveCallback;
        if (request.intermediateTimesteps > 0) {
            saveCallback = (FloatTensor f, Integer i) -> {
                try {
                    if (i % request.intermediateTimesteps == 0) {
                        var image = vae.decodeToBufferedImage(f.copy(), latentScalar);
                        System.out.println("Saving at index " + i);
                        boolean[] valid = new boolean[image.size()];
                        Arrays.fill(valid, true);
                        var sdImages = wrap(image, request, valid);
                        for (int j = 0; j < sdImages.size(); j++) {
                            save(sdImages.get(j), request.intermediatePath.resolve("batch-" + j + "-timestep-" + i + ".png"));
                        }
                    }
                } catch (OrtException | IOException e) {
                    throw new RuntimeException(e);
                }
            };
        } else {
             saveCallback = (FloatTensor f, Integer i) -> {};
        }
        return saveCallback;
    }

    /**
     * Wraps the output from the image generation in an {@link SDImage} record.
     * @param images The generated images.
     * @param request The generation request.
     * @param isValid Is this a valid image?
     * @return A list of SDImages.
     */
    private List<SDImage> wrap(List<BufferedImage> images, Request request, boolean[] isValid) {
        List<SDImage> output = new ArrayList<>();

        for (int i = 0; i < images.size(); i++) {
            output.add(new SDImage(images.get(i), modelName, request, i, isValid[i]));
        }

        return output;
    }

    @Override
    public void close() {
        try {
            embedder.close();
            unet.close();
            vae.close();
            if (safety != null) {
                safety.close();
            }
        } catch (OrtException e) {
            throw new IllegalStateException("Failed to close sessions.", e);
        }
    }

    /**
     * Constructs a SD4J CPU pipeline from the supplied model path.
     * <p>
     * Expects the following directory structure:
     * <ul>
     *     <li>VAE - $initialPath/vae_decoder/model.onnx</li>
     *     <li>Text Encoder - $initialPath/text_encoder/model.onnx</li>
     *     <li>UNet - $initialPath/unet/model.onnx</li>
     *     <li>Safety checker - $initialPath/safety_checker/model.onnx</li>
     *     <li>Tokenizer - $pwd/text_tokenizer/custom_op_cliptok.onnx</li>
     * </ul>
     * @param initialPath The path to the set of models.
     * @return The SD4J pipeline running on CPUs.
     */
    public static SD4J factory(String initialPath) {
        return factory(initialPath, false);
    }

    /**
     * Constructs a SD4J pipeline from the supplied model path, optionally on GPUs.
     * <p>
     * Expects the following directory structure:
     * <ul>
     *     <li>VAE - $initialPath/vae_decoder/model.onnx</li>
     *     <li>Text Encoder - $initialPath/text_encoder/model.onnx</li>
     *     <li>UNet - $initialPath/unet/model.onnx</li>
     *     <li>Safety checker - $initialPath/safety_checker/model.onnx</li>
     *     <li>Tokenizer - $pwd/text_tokenizer/custom_op_cliptok.onnx</li>
     * </ul>
     * @param initialPath The path to the set of models.
     * @param useCUDA Should the text encoder, unet, vae and safety checker be run on GPU?
     * @return The SD4J pipeline.
     */
    public static SD4J factory(String initialPath, boolean useCUDA) {
        return factory(new SD4JConfig(initialPath, useCUDA ? ExecutionProvider.CUDA : ExecutionProvider.CPU, 0, ModelType.SD1_5));
    }

    /**
     * Constructs a SD4J pipeline from the supplied model path, optionally on GPUs.
     * <p>
     * Expects the following directory structure:
     * <ul>
     *     <li>VAE - $initialPath/vae_decoder/model.onnx</li>
     *     <li>Text Encoder - $initialPath/text_encoder/model.onnx</li>
     *     <li>Text Encoder XL - $initialPath/text_encoder_2/model.onnx</li>
     *     <li>UNet - $initialPath/unet/model.onnx</li>
     *     <li>Safety checker - $initialPath/safety_checker/model.onnx</li>
     *     <li>Tokenizer - $pwd/text_tokenizer/custom_op_cliptok.onnx</li>
     * </ul>
     * @param config The SD4J configuration.
     * @return The SD4J pipeline.
     */
    public static SD4J factory(SD4JConfig config) {
        var rootPath = Path.of(config.modelPath());
        var modelName = rootPath.getName(rootPath.getNameCount()-1).toString();
        var vaePath = rootPath.resolve("vae_decoder/model.onnx");
        var encoderPath = rootPath.resolve("text_encoder/model.onnx");
        var encoderXLPath = rootPath.resolve("text_encoder_2/model.onnx");
        var unetPath = rootPath.resolve("unet/model.onnx");
        var safetyPath = rootPath.resolve("safety_checker/model.onnx");
        var tokenizerPath = Path.of("text_tokenizer/custom_op_cliptok.onnx");

        try {
            // Initialize the library
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            env.setTelemetry(false);
            final int deviceId = config.id();
            Supplier<OrtSession.SessionOptions> cpuSupplier = () -> {
                try {
                    var opts = new OrtSession.SessionOptions();
                    opts.setInterOpNumThreads(0);
                    opts.setIntraOpNumThreads(0);
                    return opts;
                } catch (OrtException e) {
                    throw new IllegalStateException("Failed to construct session options", e);
                }
            };
            logger.info("Config provider: " + config.provider);
            Supplier<OrtSession.SessionOptions> optsSupplier = switch (config.provider) {
                case CUDA -> () -> {
                    try {
                        var opts = new OrtSession.SessionOptions();
                        var cudaOpts = new OrtCUDAProviderOptions(deviceId);
                        cudaOpts.add("arena_extend_strategy","kSameAsRequested");
                        cudaOpts.add("cudnn_conv_algo_search","DEFAULT");
                        cudaOpts.add("do_copy_in_default_stream","1");
                        cudaOpts.add("cudnn_conv_use_max_workspace","1");
                        cudaOpts.add("cudnn_conv1d_pad_to_nc1d","1");
                        opts.addCUDA(cudaOpts);
                        return opts;
                    } catch (OrtException e) {
                        throw new IllegalStateException("Failed to create options.", e);
                    }
                };
                case CORE_ML -> () -> {
                    try {
                        var opts = new OrtSession.SessionOptions();
                        opts.setInterOpNumThreads(0);
                        opts.setIntraOpNumThreads(0);
                        opts.addCoreML(EnumSet.of(CoreMLFlags.CREATE_MLPROGRAM, CoreMLFlags.ENABLE_ON_SUBGRAPH));
                        return opts;
                    } catch (OrtException e) {
                        throw new IllegalStateException("Failed to construct session options", e);
                    }
                };
                case DIRECT_ML -> () -> {
                    try {
                        var opts = new OrtSession.SessionOptions();
                        opts.setInterOpNumThreads(0);
                        opts.setIntraOpNumThreads(0);
                        opts.addDirectML(deviceId);
                        return opts;
                    } catch (OrtException e) {
                        throw new IllegalStateException("Failed to construct session options", e);
                    }
                };
                case OPENVINO -> () -> {
                    if (!OrtEnvironment.getAvailableProviders().contains(OrtProvider.OPEN_VINO)) {
                        throw new IllegalStateException("OpenVINO provider not available; available providers: " + OrtEnvironment.getAvailableProviders());
                    }
                    try {
                        var opts = new OrtSession.SessionOptions();
                        opts.setInterOpNumThreads(0);
                        opts.setIntraOpNumThreads(0);
                        opts.addOpenVINO("AUTO:CPU,NPU,GPU");
                        return opts;
                    } catch (OrtException e) {
                        throw new IllegalStateException("Failed to construct session options", e);
                    }
                };
                case CPU -> cpuSupplier;
            };
            TextEmbedder embedder = new TextEmbedder(tokenizerPath, encoderPath, optsSupplier.get(), config.type.textDimSize, false);
            logger.info("Loaded embedder from " + encoderPath);
            TextEmbedder embedderXL = null;
            if (config.type == ModelType.SDXL) {
                embedderXL = new TextEmbedder(tokenizerPath, encoderXLPath, optsSupplier.get(), config.type.text2DimSize, true);
                logger.info("Loaded second embedder from " + encoderXLPath);
            }
            UNet unet = new UNet(unetPath, optsSupplier.get());
            logger.info("Loaded unet from " + unetPath);
            VAEDecoder vae = new VAEDecoder(vaePath, optsSupplier.get());
            logger.info("Loaded vae from " + vaePath);
            SafetyChecker safety;
            if (safetyPath.toFile().exists()) {
                safety = new SafetyChecker(safetyPath, optsSupplier.get());
                logger.info("Created safety");
            } else {
                safety = null;
                logger.info("No safety found");
            }
            return new SD4J(modelName, embedder, embedderXL, unet, vae, safety);
        } catch (OrtException e) {
            throw new IllegalStateException("Failed to instantiate SD4J pipeline", e);
        }
    }

    /**
     * An image generated from Stable Diffusion, along with the input text and other inference properties.
     * @param image The image.
     * @param modelName The model name.
     * @param request The image generation request.
     * @param batchId The id number within the batch.
     * @param isValid Did the image pass the safety check?
     */
    public record SDImage(BufferedImage image, String modelName, Request request, int batchId, boolean isValid) {
        public String metadataDescription() {
            StringBuilder sb = new StringBuilder();

            sb.append(request.text());
            sb.append('\n');
            if (!request.negText().isEmpty()) {
                sb.append("Negative prompt: ");
                sb.append(request.negText());
                sb.append('\n');
            }
            sb.append("Steps: ");
            sb.append(request.steps());
            sb.append(", ");
            sb.append("Sampler: ");
            sb.append(request.scheduler().descriptionName());
            sb.append(", ");
            sb.append("CFG scale: ");
            sb.append(request.guidance());
            sb.append(", ");
            sb.append("Seed: ");
            sb.append(request.seed());
            sb.append(", ");
            sb.append("Size: ");
            sb.append(request.size().width());
            sb.append("x");
            sb.append(request.size().height());
            sb.append(", ");
            sb.append("Model: ");
            sb.append(modelName);
            sb.append(", ");
            sb.append("Batch id: ");
            sb.append(batchId);
            return sb.toString();
        }
    }

    /**
     * Image size.
     * @param height Height in pixels.
     * @param width Width in pixels.
     */
    public record ImageSize(int height, int width) {
        /**
         * Creates a square image size.
         * @param size The height and width.
         */
        public ImageSize(int size) {
            this(size, size);
        }

        @Override
        public String toString() {
            return "[" + height + ", " + width + "]";
        }
    }

    /**
     * A image creation request.
     * @param text The image text.
     * @param negText The image negative text.
     * @param steps The number of diffusion steps.
     * @param guidance The strength of the classifier-free guidance.
     * @param seed The RNG seed used to initialize the image (and any ancestral sampling noise).
     * @param size The requested image size.
     * @param scheduler The scheduling algorithm.
     * @param batchSize The batch size.
     * @param intermediatePath The path to output the intermediate denoising images to.
     * @param intermediateTimesteps The number of steps between each intermediate image.
     */
    public record Request(String text, String negText, int steps, float guidance, int seed, ImageSize size, Schedulers scheduler, int batchSize, Path intermediatePath, int intermediateTimesteps) {
        public Request {
            if (intermediateTimesteps > 0 && intermediatePath == null) {
                throw new IllegalArgumentException("Intermediate path required if intermediate timesteps is positive.");
            }
            if (intermediateTimesteps > steps) {
                logger.warning("Intermediate timesteps must be less than steps, no intermediates will be generated.");
            }
            try {
                if (intermediatePath != null) {
                    Files.createDirectory(intermediatePath);
                }
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        Request(String text, String negText, String stepsStr, String guidanceStr, String seedStr, ImageSize size, Schedulers scheduler, String batchSize) {
            this(text.strip(), negText.strip(), Integer.parseInt(stepsStr), Float.parseFloat(guidanceStr), Integer.parseInt(seedStr), size, scheduler, Integer.parseInt(batchSize), null, -1);
        }
        Request(String text, String negText, String stepsStr, String guidanceStr, String seedStr, ImageSize size, Schedulers scheduler, String batchSize, String intermediatePath, String intermediateTimesteps) {
            this(text.strip(), negText.strip(), Integer.parseInt(stepsStr), Float.parseFloat(guidanceStr), Integer.parseInt(seedStr), size, scheduler, Integer.parseInt(batchSize), Path.of(intermediatePath), Integer.parseInt(intermediateTimesteps));
        }
    }

    /**
     * Supported execution providers.
     */
    public enum ExecutionProvider {
        /**
         * CPU.
         */
        CPU,
        /**
         * Apple's Core ML.
         */
        CORE_ML,
        /**
         * Nvidia GPUs.
         */
        CUDA,
        /**
         * OpenVINO.
         */
        OPENVINO,
        /**
         * Windows DirectML devices.
         */
        DIRECT_ML;

        /**
         * Looks up an execution provider returning the enum or throwing {@link IllegalArgumentException} if it's unknown.
         * @param name The ep to lookup.
         * @return The enum value.
         */
        public static ExecutionProvider lookup(String name) {
            String lower = name.toLowerCase(Locale.US);
            return switch (lower) {
                case "cpu", "" -> CPU;
                case "coreml", "core_ml", "core-ml" -> CORE_ML;
                case "cuda" -> CUDA;
                case "openvino" -> OPENVINO;
                case "directml", "direct_ml", "direct-ml" -> DIRECT_ML;
                default -> { throw new IllegalArgumentException("Unknown execution provider '" + name + "'"); }
            };
        }
    }

    /**
     * The type of Stable Diffusion model.
     */
    public enum ModelType {
        SD1_5(TextEmbedder.SD_1_5_DIM_SIZE,-1),
        SD2(TextEmbedder.SD_2_DIM_SIZE,-1),
        SDXL(TextEmbedder.SD_1_5_DIM_SIZE,TextEmbedder.SDXL_DIM_SIZE);

        /**
         * The text dimension size for the first encoder.
         */
        public final int textDimSize;
        /**
         * The text dimension size for the second encoder.
         */
        public final int text2DimSize;

        private ModelType(int textDimSize, int text2DimSize) {
            this.textDimSize = textDimSize;
            this.text2DimSize = text2DimSize;
        }

        /**
         * Looks up the model type returning the enum or throwing {@link IllegalArgumentException} if it's unknown.
         * @param name The model type to lookup.
         * @return The enum value.
         */
        public static ModelType lookup(String name) {
            String lower = name.toLowerCase(Locale.US);
            return switch (lower) {
                case "sdv1.5", "sd15", "sd1.5", "sd1_5", "sd1", "sdv1" -> SD1_5;
                case "sdv2", "sdv21", "sdv2.1", "sd-turbo", "sd_turbo" -> SD2;
                case "sdxl", "sdxl-turbo", "sdxl_turbo" -> SDXL;
                default -> { throw new IllegalArgumentException("Unknown model type '" + name + "'"); }
            };
        }
    }

    /**
     * Record for the SD4J configuration.
     * @param modelPath The path to the onnx models.
     * @param provider The execution provider to use.
     * @param id The device id.
     */
    public record SD4JConfig(String modelPath, ExecutionProvider provider, int id, ModelType type) {
        /**
         * Parses the arguments into a config.
         * @param args The arguments.
         * @return A SD4J config.
         */
        public static Optional<SD4JConfig> parseArgs(String[] args) {
            String modelPath = "";
            String ep = "";
            String modelType = "sd1.5";
            int id = 0;
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--help", "--usage" -> {
                        return Optional.empty();
                    }
                    case "--model-path", "-p" -> {
                        // check if there's another argument, otherwise return empty
                        if (i == args.length - 1) {
                            // No model path
                            return Optional.empty();
                        } else {
                            // Consume argument
                            i++;
                            modelPath = args[i];
                        }
                    }
                    case "--execution-provider", "--ep" -> {
                        // check if there's another argument, otherwise return empty
                        if (i == args.length - 1) {
                            // No provider
                            return Optional.empty();
                        } else {
                            // Consume argument
                            i++;
                            ep = args[i];
                        }
                    }
                    case "--device-id" -> {
                        // check if there's another argument, otherwise return empty
                        if (i == args.length - 1) {
                            // No id
                            return Optional.empty();
                        } else {
                            // Consume argument
                            i++;
                            id = Integer.parseInt(args[i]);
                        }
                    }
                    case "--model-type", "-m" -> {
                        // check if there's another argument, otherwise return empty
                        if (i == args.length - 1) {
                            // No provider
                            return Optional.empty();
                        } else {
                            // Consume argument
                            i++;
                            modelType = args[i];
                        }
                    }
                    default -> {
                        // Unexpected argument
                        logger.warning("Unexpected argument '" + args[i] + "'");
                        return Optional.empty();
                    }
                }
            }
            return Optional.of(new SD4JConfig(modelPath, ExecutionProvider.lookup(ep), id, ModelType.lookup(modelType)));
        }

        /**
         * Help string for the config arguments.
         * @return The help string.
         */
        public static String help() {
            return "SD4J --model-path <model-path> --execution-provider {CUDA,CoreML,DirectML,CPU,OPENVINO} (optional --device-id <int> --model-type <sd1.5,sd2,sdxl>)";
        }
    }
}
