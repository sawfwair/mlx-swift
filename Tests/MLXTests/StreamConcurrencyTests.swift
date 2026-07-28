import MLX
import XCTest

private actor StreamConcurrencyHarness {
    func evaluateGraph() async -> [Float] {
        await Stream.withNewDefaultStream {
            let input = MLXArray([Float32(1), 2, 3])
            let output = input * 2
            await Task.yield()
            MLX.eval(output)
            return output.asArray(Float.self)
        }
    }
}

final class StreamConcurrencyTests: XCTestCase {
    func testAsyncDefaultStreamGraphCanEvaluateOnDetachedThread() async {
        let graph = await Stream.withNewDefaultStream {
            let input = MLXArray([Float32(1), 2, 3])
            let output = input * 2
            await Task.yield()
            return output
        }

        let values = await Task.detached {
            MLX.eval(graph)
            return graph.asArray(Float.self)
        }.value

        XCTAssertEqual(values, [2, 4, 6])
    }

    func testAsyncDefaultStreamSupportsExplicitCPUAndGPUAfterExecutorHop() async {
        let graphs = await Stream.withNewDefaultStream {
            let cpuOutput = multiply(
                ones([3], type: Float.self, stream: .cpu),
                2,
                stream: .cpu
            )
            let gpuOutput = multiply(
                ones([3], type: Float.self, stream: .gpu),
                3,
                stream: .gpu
            )
            await Task.yield()
            return (cpuOutput, gpuOutput)
        }

        let values = await Task.detached {
            MLX.eval(graphs.0, graphs.1)
            return (
                graphs.0.asArray(Float.self),
                graphs.1.asArray(Float.self)
            )
        }.value

        XCTAssertEqual(values.0, [2, 2, 2])
        XCTAssertEqual(values.1, [3, 3, 3])
    }

    #if os(Linux)
        func testAsyncCPUStreamDoesNotRequireGPU() async {
            let graph = await Stream.withNewDefaultStream(device: .cpu) {
                let output = multiply(
                    ones([3], type: Float.self, stream: .cpu),
                    4,
                    stream: .cpu
                )
                await Task.yield()
                return output
            }

            let values = await Task.detached {
                MLX.eval(graph)
                return graph.asArray(Float.self)
            }.value

            XCTAssertEqual(values, [4, 4, 4])
        }
    #endif

    func testAsyncDefaultStreamPreservesCallerActorIsolation() async {
        let values = await StreamConcurrencyHarness().evaluateGraph()
        XCTAssertEqual(values, [2, 4, 6])
    }
}
