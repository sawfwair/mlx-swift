// Regression guard for the generation-17 NAX steel-gemm/qmm miscompute.
//
// The owned MLX fork gates NAX to GPU generation 18+ because the generation-17
// NAX matmul path returns wrong results for fp16 GEMM at M >= 8, N >= 8192 (and
// quantized matmul at M >= 64, N >= 9216). If that gate regresses, this shape
// diverges sharply from the fp32 reference.

import Foundation
import MLX
import MLXRandom
import XCTest

class NAXGateRegressionTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testNAXThresholdMatmulMatchesReference() {
        MLXRandom.seed(0)
        let m = 16
        let k = 512
        let n = 16384

        let a = MLXRandom.normal([m, k])
        let b = MLXRandom.normal([k, n])

        let reference = a.matmul(b)
        let actual = a.asType(.float16).matmul(b.asType(.float16)).asType(.float32)

        let difference = actual - reference
        let differenceFrobenius = (difference * difference).sum().sqrt().item(Float.self)
        let referenceFrobenius = (reference * reference).sum().sqrt().item(Float.self)
        let relativeError = differenceFrobenius / referenceFrobenius

        XCTAssertLessThan(
            relativeError,
            0.05,
            "fp16 NAX-threshold matmul relative error \(relativeError) exceeds 0.05; "
                + "check the generation-18 gate in is_nax_available()"
        )
    }
}
