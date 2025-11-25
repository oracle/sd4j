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

import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.BorderFactory;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;
import javax.swing.border.Border;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.Container;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.event.ActionEvent;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A Swing app for generating images using SD4J pipelines.
 */
public final class SD4JApp extends JFrame {
    private static final Logger logger = Logger.getLogger(SD4JApp.class.getName());

    /**
     * The SD4J pipeline.
     */
    private final SD4J diff;

    private final JPanel panel;

    /**
     * Creates a SD4JApp panel and pipeline.
     *
     * @param config The SD4J configuration.
     */
    public SD4JApp(SD4J.SD4JConfig config) {
        diff = SD4J.factory(config);

        setTitle("SD4J");

        Supplier<Border> border10px = () -> BorderFactory.createEmptyBorder(10,10,10,10);
        Supplier<Border> border5px = () -> BorderFactory.createEmptyBorder(5,5,5,5);

        GridBagLayout layout = new GridBagLayout();
        panel = new JPanel();
        panel.setLayout(layout);
        panel.setBorder(border10px.get());
        GridBagConstraints constraints = new GridBagConstraints();
        constraints.ipadx = 10;
        constraints.ipady = 10;
        constraints.fill = GridBagConstraints.HORIZONTAL;
        constraints.weightx = 1.0;
        constraints.weighty = 1.5;

        JLabel textLbl = new JLabel("Image text:");
        constraints.gridx = 0;
        constraints.gridy = 0;
        textLbl.setBorder(border5px.get());
        panel.add(textLbl, constraints);
        JTextArea textArea = new JTextArea();
        textArea.setLineWrap(true);
        textArea.setRows(3);
        constraints.gridx = 1;
        constraints.gridy = 0;
        textArea.setBorder(border5px.get());
        panel.add(textArea, constraints);

        JLabel negLbl = new JLabel("Image negative text:");
        constraints.gridx = 0;
        constraints.gridy = 1;
        negLbl.setBorder(border5px.get());
        panel.add(negLbl, constraints);
        JTextArea negArea = new JTextArea();
        negArea.setLineWrap(true);
        negArea.setRows(3);
        constraints.gridx = 1;
        constraints.gridy = 1;
        negArea.setBorder(border5px.get());
        panel.add(negArea, constraints);

        JLabel guidanceLbl = new JLabel("Guidance scale:");
        constraints.gridx = 0;
        constraints.gridy = 2;
        guidanceLbl.setBorder(border5px.get());
        panel.add(guidanceLbl, constraints);
        JTextField guidanceField = new JTextField("7.5");
        constraints.gridx = 1;
        constraints.gridy = 2;
        guidanceField.setBorder(border5px.get());
        panel.add(guidanceField, constraints);

        JLabel seedLbl = new JLabel("Seed:");
        constraints.gridx = 0;
        constraints.gridy = 3;
        seedLbl.setBorder(border5px.get());
        panel.add(seedLbl, constraints);
        JTextField seedField = new JTextField("42");
        constraints.gridx = 1;
        constraints.gridy = 3;
        seedField.setBorder(border5px.get());
        panel.add(seedField, constraints);

        JLabel stepsLbl = new JLabel("Inference steps:");
        constraints.gridx = 0;
        constraints.gridy = 4;
        stepsLbl.setBorder(border5px.get());
        panel.add(stepsLbl, constraints);
        JTextField stepsField = new JTextField("5");
        constraints.gridx = 1;
        constraints.gridy = 4;
        stepsField.setBorder(border5px.get());
        panel.add(stepsField, constraints);

        JLabel sizeLbl = new JLabel("Image size:");
        constraints.gridx = 0;
        constraints.gridy = 5;
        sizeLbl.setBorder(border5px.get());
        panel.add(sizeLbl, constraints);
        var options = new SD4J.ImageSize[]{
                new SD4J.ImageSize(256),
                new SD4J.ImageSize(512, 256),
                new SD4J.ImageSize(256, 512),
                new SD4J.ImageSize(512),
                new SD4J.ImageSize(768, 512),
                new SD4J.ImageSize(512, 768),
                new SD4J.ImageSize(768),
                new SD4J.ImageSize(1024, 768),
                new SD4J.ImageSize(768, 1024),
                new SD4J.ImageSize(1024)
        };
        JComboBox<SD4J.ImageSize> size = new JComboBox<>(options);
        size.setSelectedItem(options[3]);
        constraints.gridx = 1;
        constraints.gridy = 5;
        size.setBorder(border5px.get());
        panel.add(size, constraints);

        JLabel schedulerLbl = new JLabel("Image scheduler:");
        constraints.gridx = 0;
        constraints.gridy = 6;
        schedulerLbl.setBorder(border5px.get());
        panel.add(schedulerLbl, constraints);
        var schedOptions = new Schedulers[]{
                Schedulers.LMS,
                Schedulers.EULER_ANCESTRAL
        };
        JComboBox<Schedulers> scheduler = new JComboBox<>(schedOptions);
        scheduler.setSelectedItem(Schedulers.LMS);
        constraints.gridx = 1;
        constraints.gridy = 6;
        scheduler.setBorder(border5px.get());
        panel.add(scheduler, constraints);

        JLabel batchLbl = new JLabel("Batch Size:");
        constraints.gridx = 0;
        constraints.gridy = 7;
        batchLbl.setBorder(border5px.get());
        panel.add(batchLbl, constraints);
        JTextField batchField = new JTextField("1");
        constraints.gridx = 1;
        constraints.gridy = 7;
        batchField.setBorder(border5px.get());
        panel.add(batchField, constraints);

        JLabel pathLbl = new JLabel("Intermediate steps path:");
        constraints.gridx = 0;
        constraints.gridy = 8;
        pathLbl.setBorder(border5px.get());
        panel.add(pathLbl, constraints);
        JTextField pathField = new JTextField("");
        constraints.gridx = 1;
        constraints.gridy = 8;
        pathField.setBorder(border5px.get());
        panel.add(pathField, constraints);

        JLabel interStepsLbl = new JLabel("Intermediate step freq:");
        constraints.gridx = 0;
        constraints.gridy = 9;
        interStepsLbl.setBorder(border5px.get());
        panel.add(interStepsLbl, constraints);
        JTextField interStepsField = new JTextField("-1");
        constraints.gridx = 1;
        constraints.gridy = 9;
        interStepsField.setBorder(border5px.get());
        panel.add(interStepsField, constraints);

        JButton btn = new JButton("Generate");
        btn.addActionListener((ActionEvent e) -> {
            SD4J.Request request;
            if (pathField.getText().isEmpty()) {
                request = new SD4J.Request(textArea.getText(), negArea.getText(), stepsField.getText(),
                        guidanceField.getText(), seedField.getText(), (SD4J.ImageSize) size.getSelectedItem(), (Schedulers) scheduler.getSelectedItem(),
                        batchField.getText());
            } else {
                request = new SD4J.Request(textArea.getText(), negArea.getText(), stepsField.getText(),
                        guidanceField.getText(), seedField.getText(), (SD4J.ImageSize) size.getSelectedItem(), (Schedulers) scheduler.getSelectedItem(),
                        batchField.getText(), pathField.getText(), interStepsField.getText());
            }
            SwingDisplayWindow.create(diff, request);
        });
        constraints.gridx = 1;
        constraints.gridy = 10;
        btn.setBorder(border5px.get());
        panel.add(btn, constraints);

        setSize(700, 425);

        setDefaultCloseOperation(EXIT_ON_CLOSE);

        add(panel);
        panel.setVisible(true);
    }

    /**
     * A holding class for the generation thread and the window for this image.
     */
    static final class SwingDisplayWindow extends JFrame {
        Thread computation;
        JFileChooser chooser;

        SwingDisplayWindow() {
            chooser = new JFileChooser();
            FileNameExtensionFilter filter = new FileNameExtensionFilter(
                    "PNG Images", "png");
            chooser.setFileFilter(filter);
        }

        /**
         * Sets the computation thread, must only be called once.
         *
         * @param computation The computation thread.
         */
        private void setThread(Thread computation) {
            this.computation = computation;
        }

        /**
         * Draws a progress bar on the window.
         *
         * @param request The computation request to link up the image.
         * @return A consumer which updates the progress bar for each step of diffusion.
         */
        private Consumer<Integer> drawProgress(SD4J.Request request) {
            setTitle("Generating Image...");
            Container contents = getContentPane();
            contents.setLayout(new GridBagLayout());
            GridBagConstraints constraints = new GridBagConstraints();
            constraints.fill = GridBagConstraints.NONE;
            constraints.weightx = 1.0;
            constraints.weighty = 1.5;

            JLabel progressLbl = new JLabel("Generation progress:");
            constraints.anchor = GridBagConstraints.CENTER;
            constraints.gridx = 0;
            constraints.gridy = 0;
            contents.add(progressLbl, constraints);

            JProgressBar progressBar = new JProgressBar(0, request.steps());
            constraints.anchor = GridBagConstraints.CENTER;
            constraints.gridx = 1;
            constraints.gridy = 0;
            contents.add(progressBar, constraints);
            setSize(request.size().width() + 100, request.size().height() + 100);

            contents.setVisible(true);

            return progressBar::setValue;
        }

        /**
         * Draws the image onto the window.
         *
         * @param image The generated image.
         */
        private void drawImage(SD4J.SDImage image) {
            setTitle("Stable Diffusion Image");

            Container contents = getContentPane();
            contents.setVisible(false);
            contents.removeAll();
            GridBagConstraints constraints = new GridBagConstraints();
            constraints.fill = GridBagConstraints.BOTH;
            constraints.weightx = 1.0;
            constraints.weighty = 1.0;

            constraints.anchor = GridBagConstraints.CENTER;
            constraints.gridx = 0;
            constraints.gridy = 0;
            JLabel imageLbl = new JLabel(new ImageIcon(image.image()));

            imageLbl.setSize(image.image().getWidth(), image.image().getHeight());
            contents.add(imageLbl, constraints);

            constraints.anchor = GridBagConstraints.CENTER;
            constraints.gridx = 0;
            constraints.gridy = 1;
            JButton btn = new JButton("Save");
            btn.addActionListener((ActionEvent e) -> {
                int returnVal = chooser.showSaveDialog(this);
                if (returnVal == JFileChooser.APPROVE_OPTION) {
                    File file = chooser.getSelectedFile();
                    if (!file.getName().endsWith(".png")) {
                        file = new File(file + ".png");
                    }
                    logger.info("Saving file to " + file.getName());
                    try {
                        SD4J.save(image, file);
                    } catch (IOException ex) {
                        logger.log(Level.WARNING, "Failed to save file", ex);
                    }
                }
            });
            contents.add(btn, constraints);

            setSize(image.image().getWidth() + 100, image.image().getHeight() + 100);

            pack();
            contents.setVisible(true);
        }

        /**
         * Writes an error message on the window.
         *
         * @param request The request which generated an invalid image.
         */
        private void drawInvalid(SD4J.Request request) {
            setTitle("Invalid image generated");

            Container contents = getContentPane();
            contents.setVisible(false);
            contents.removeAll();
            GridBagConstraints constraints = new GridBagConstraints();
            constraints.fill = GridBagConstraints.BOTH;
            constraints.weightx = 1.0;
            constraints.weighty = 1.0;

            constraints.anchor = GridBagConstraints.CENTER;
            constraints.gridx = 0;
            constraints.gridy = 0;
            JLabel lbl = new JLabel("Invalid image generated for text '" + request.text() + "', and negative text '" + request.negText() + "'");
            contents.add(lbl, constraints);

            setSize(request.size().width() + 100, request.size().height() + 100);
            pack();
            contents.setVisible(true);
        }

        /**
         * Creates a window and triggers the image generation in a separate thread.
         *
         * @param diffusion The SD4J pipeline.
         * @param request   The generation request.
         */
        static void create(SD4J diffusion, SD4J.Request request) {
            List<SwingDisplayWindow> windows = new ArrayList<>(request.batchSize());
            List<Consumer<Integer>> callbacks = new ArrayList<>(request.batchSize());

            for (int i = 0; i < request.batchSize(); i++) {
                SwingDisplayWindow window = new SwingDisplayWindow();
                windows.add(window);
                Consumer<Integer> progressCallback = window.drawProgress(request);
                callbacks.add(progressCallback);
            }
            Consumer<Integer> progressCallback = (i) -> {
                for (var callback : callbacks) {
                    callback.accept(i);
                }
            };

            Runnable r = () -> {
                var images = diffusion.generateImage(request, progressCallback);
                for (int i = 0; i < request.batchSize(); i++) {
                    var image = images.get(i);
                    final var finalI = i;
                    if (image.isValid()) {
                        SwingUtilities.invokeLater(() -> windows.get(finalI).drawImage(image));
                    } else {
                        SwingUtilities.invokeLater(() -> windows.get(finalI).drawInvalid(request));
                    }
                }
            };
            Thread t = new Thread(r);
            t.setDaemon(true);
            for (int i = 0; i < request.batchSize(); i++) {
                windows.get(i).setThread(t);
                windows.get(i).setVisible(true);
            }
            t.start();
        }
    }

    /**
     * Swing app entry point.
     *
     * @param args The CLI args.
     */
    public static void main(String[] args) throws UnsupportedLookAndFeelException, ClassNotFoundException, InstantiationException, IllegalAccessException {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        Optional<SD4J.SD4JConfig> config = SD4J.SD4JConfig.parseArgs(args);
        if (config.isPresent()) {
            SD4JApp gui = new SD4JApp(config.get());
            gui.setVisible(true);
        } else {
            System.out.println(SD4J.SD4JConfig.help());
            System.exit(1);
        }
    }
}
