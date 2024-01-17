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

import java.util.function.Function;

/**
 * The set of available scheduler algorithms.
 */
public enum Schedulers {
    /**
     * A Linear Multi-Step scheduler.
     */
    LMS(a -> new LMSDiscreteScheduler(), "LMS", "LMS"),
    /**
     * An Euler Ancestral scheduler.
     */
    EULER_ANCESTRAL(EulerAncestralDiscreteScheduler::new, "Euler Ancestral", "Euler a");

    final Function<Long, Scheduler> factory;
    final String displayName;
    final String descriptionName;

    Schedulers(Function<Long,Scheduler> factory, String displayName, String descriptionName) {
        this.factory = factory;
        this.displayName = displayName;
        this.descriptionName = descriptionName;
    }

    @Override
    public String toString() {
        return displayName;
    }

    /**
     * The name to be used in the image metadata.
     * @return The image metadata scheduler name.
     */
    public String descriptionName() {
        return descriptionName;
    }

    /**
     * Creates a scheduler and supplies a seed for RNG.
     * @param seed The RNG seed.
     * @return The scheduler object.
     */
    public Scheduler create(long seed) {
        return factory.apply(seed);
    }
}
