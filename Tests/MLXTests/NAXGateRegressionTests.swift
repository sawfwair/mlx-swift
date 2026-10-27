// Floating-point NAX stays available; quantized dispatch owns its workaround.

import Foundation
import MLX
import XCTest

class NAXGateRegressionTests: XCTestCase {

    override class func setUp() {
    }

    func testNAXThresholdMatmulMatchesReference() {
        MLXRandom.seed(0)
        let m = 16
        let k = 512
        let n = 16384

        let a = MLXRandom.normal([m, k])
        let b = MLXRandom.normal([k, n])

        let reference = a.matmul(b)
        for dtype: DType in [.float16, .bfloat16] {
            let actual = a.asType(dtype).matmul(b.asType(dtype)).asType(.float32)
            let difference = actual - reference
            let differenceFrobenius = (difference * difference).sum().sqrt().item(Float.self)
            let referenceFrobenius = (reference * reference).sum().sqrt().item(Float.self)
            let relativeError = differenceFrobenius / referenceFrobenius

            XCTAssertLessThan(
                relativeError,
                0.05,
                "\(dtype) NAX-threshold matmul relative error \(relativeError) exceeds 0.05"
            )
        }
    }

    func testQuantizedNAXThresholdMatchesReference() {
        MLXRandom.seed(1729)
        for dtype: DType in [.float16, .bfloat16] {
            for bits in [1, 2, 4, 8] {
                let x = MLXRandom.normal([64, 512]).asType(dtype)
                let w = MLXRandom.normal([9216, 512]).asType(dtype)
                let (packed, scales, biases) = quantized(w, groupSize: 128, bits: bits)
                let restored = dequantized(
                    packed, scales: scales, biases: biases, groupSize: 128,
                    bits: bits, dtype: .float32)
                let reference = matmul(x.asType(.float32), restored.T)
                let actual = quantizedMM(
                    x, packed, scales: scales, biases: biases,
                    groupSize: 128, bits: bits
                ).asType(.float32)
                let difference = actual - reference
                let relativeError =
                    ((difference * difference).sum() / (reference * reference).sum()).sqrt().item(
                        Float.self)
                XCTAssertLessThan(
                    relativeError, 1e-2,
                    "\(dtype) \(bits)-bit threshold matmul relative error \(relativeError)")
            }
        }
    }

}
