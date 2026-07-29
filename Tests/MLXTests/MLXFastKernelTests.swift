// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class MLXFastKernelTests: XCTestCase {

    func testCustomKernelBasic() {
        // based on def test_custom_kernel_basic
        MLXRandom.seed(7)
        let a = normal([2, 2])
        let kernel = MLXFast.metalKernel(
            name: "basic",
            inputNames: ["a"],
            outputNames: ["out1"],
            source: """
                    uint elem = thread_position_in_grid.x;
                    out1[elem] = a[elem];
                """)

        let out = kernel(
            [a],
            grid: (4, 1, 1),
            threadGroup: (2, 1, 1),
            outputShapes: [[2, 2]],
            outputDTypes: [.float32])

        XCTAssertTrue(allClose(out[0], a).all().item())
    }

    func testCustomKernelArgs() {
        // based on def test_custom_kernel_args
        MLXRandom.seed(7)
        let a = normal([3, 6])
        let c = normal([2, 2]).asType(.bfloat16)

        let kernel = MLXFast.metalKernel(
            name: "arg_test",
            inputNames: ["a", "b", "c", "d"],
            outputNames: ["out1", "out2"],
            source: """
                    uint elem = thread_position_in_grid.x;
                    T tmp = a[0];
                    if (e) {
                        out1[elem] = a[1] + b[2] + c[3] + d + f;
                    } else {
                        out1[elem] = 1;
                    }
                    out2[elem] = a[1] + b[2] + c[1] - d;
                """)

        let out = kernel(
            [
                a,
                MLXArray([3, 4, 5]),
                c,
                7.3,
            ],
            template: [
                ("e", true),
                ("f", 3),
                ("T", DType.float16),
            ],
            grid: (6, 1, 1),
            threadGroup: (2, 1, 1),
            outputShapes: [[2, 2], [3, 2]],
            outputDTypes: [.float32, .int32])

        XCTAssertTrue(allClose(out[0], full([2, 2], values: 14.0484)).all().item())
        XCTAssertTrue(allClose(out[1], full([3, 2], values: -2)).all().item())
    }

    func testCustomKernelFPQuantizedHeaders() {
        let kernel = MLXFast.metalKernel(
            name: "fp_quantized_headers",
            inputNames: ["input"],
            outputNames: ["output"],
            source: """
                    output[0] = get_pack_factor<8, 4>();
                """,
            header: "// MLX_INCLUDE_FP_QUANTIZED_HEADERS\n"
        )
        let output = kernel(
            [MLXArray([0])],
            grid: (1, 1, 1),
            threadGroup: (1, 1, 1),
            outputShapes: [[1]],
            outputDTypes: [.uint32]
        )[0]

        XCTAssertEqual(output.item(UInt32.self), 2)
    }

    func testNVFP4PackedDecodeMatchesScalarHeaderPath() {
        let kernel = MLXFast.metalKernel(
            name: "nvfp4_packed_decode_matches_scalar",
            inputNames: ["packed", "scales"],
            outputNames: ["mismatch"],
            source: """
                    uint gid = thread_position_in_grid.x;
                    uint8_t packed_byte = packed[gid & 255];
                    uint8_t scale_byte = scales[gid >> 8];
                    uint32_t codes = uint32_t(packed_byte) * 0x01010101u;

                    bfloat fast[8];
                    fp4nv_decode8<bfloat>(
                        codes, fp4nv_scale_x16384(scale_byte), fast);

                    bfloat scale = dequantize_scale<bfloat, 16>(scale_byte);
                    uint different = 0;
                    for (uint i = 0; i < 4; i++) {
                      bfloat low = scale * Dequantize<4, bfloat>{}(packed_byte);
                      bfloat high =
                          scale * Dequantize<4, bfloat>{}(packed_byte >> 4);
                      different |=
                          as_type<ushort>(fast[2 * i]) != as_type<ushort>(low);
                      different |= as_type<ushort>(fast[2 * i + 1]) !=
                          as_type<ushort>(high);
                    }
                    mismatch[gid] = different;
                """,
            header: "// MLX_INCLUDE_FP_QUANTIZED_HEADERS\n"
        )
        let byteValues = MLXArray((0...255).map(UInt8.init))
        let mismatches = kernel(
            [byteValues, byteValues],
            grid: (256 * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[256 * 256]],
            outputDTypes: [.uint32]
        )[0]

        XCTAssertEqual(MLX.sum(mismatches).item(UInt32.self), 0)
    }

    func testCustomKernelAffineQuantizedHeaders() {
        let kernel = MLXFast.metalKernel(
            name: "affine_quantized_headers",
            inputNames: ["input"],
            outputNames: ["output"],
            source: """
                    output[0] = get_pack_factor<8, 32>();
                """,
            header: "// MLX_INCLUDE_AFFINE_QUANTIZED_HEADERS\n"
        )
        let output = kernel(
            [MLXArray([0])],
            grid: (1, 1, 1),
            threadGroup: (1, 1, 1),
            outputShapes: [[1]],
            outputDTypes: [.uint32]
        )[0]

        XCTAssertEqual(output.item(UInt32.self), 4)
    }

    func testFastSDPA() {
        // https://github.com/ml-explore/mlx-swift/issues/172
        // this will just make sure the MLXFast.scaled_dot_product_attention is
        // callable in the various cases, based on
        // https://github.com/ml-explore/mlx/blob/main/python/tests/test_fast_sdpa.py#L65-L87

        let Dk = 64
        let scale = 1.0 / sqrt(Float(Dk))
        let dTypes = [DType.float32, DType.float16]
        for SEQUENCE_LENGTH in [63, 129, 400] {
            for dtype in dTypes {
                let B = 2
                let H = 24
                let q = MLXRandom.normal([B, H, SEQUENCE_LENGTH, Dk]).asType(dtype)
                let k = MLXRandom.normal([B, H, SEQUENCE_LENGTH, Dk]).asType(dtype)
                let v = MLXRandom.normal([B, H, SEQUENCE_LENGTH, Dk]).asType(dtype)

                let result = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: nil,
                    memoryEfficientThreshold: 2)

                eval(result)
            }
        }
    }

    func testFastSDPAOutput() {
        MLXRandom.seed(0)
        let queries = MLXRandom.uniform(0.0 ..< 1.0, [1, 32, 1, 80])
        let keys = MLXRandom.uniform(0.0 ..< 1.0, [1, 32, 9, 80])
        let values = MLXRandom.uniform(0.0 ..< 1.0, [1, 32, 9, 80])
        let result = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 0.1118, mask: .none)
        print(result.shape)
        print(result.sum().item(Float.self))
        XCTAssertEqual(result.shape, [1, 32, 1, 80])
        XCTAssertEqual(result.sum().item(Float.self), 1281.9253, accuracy: 0.01)
    }

    func testRoPEOutput() {
        // https://github.com/ml-explore/mlx-swift/issues/315
        MLXRandom.seed(0)
        let rope = RoPE(dimensions: 32, traditional: false, base: 10_000, scale: 1)
        let queries = MLXRandom.uniform(0.0 ..< 1.0, [1, 32, 1, 80])
        print(queries.shape)
        let result = rope(queries, offset: 8)
        XCTAssertEqual(result.sum().item(Float.self), 1079.7894, accuracy: 0.01)
    }
}
