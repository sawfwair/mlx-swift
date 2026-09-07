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
    func testReusableContextRetainsDefaultAndExplicitStreamsAfterSuspension() async {
        let context = MLX.Stream.Context()
        var previous: (MLX.Stream, MLX.Stream, MLX.Stream)?
        for _ in 0..<100 {
            let streams = await Stream.withDefaultStream(context) {
                let streams = (
                    StreamOrDevice.default.stream,
                    StreamOrDevice.cpu.stream,
                    StreamOrDevice.gpu.stream
                )
                let cpu = multiply(ones([3], stream: .cpu), 2, stream: .cpu)
                let gpu = multiply(ones([3], stream: .gpu), 3, stream: .gpu)
                await Task.yield()
                XCTAssertEqual(StreamOrDevice.default.stream, streams.0)
                XCTAssertEqual(cpu.asArray(Float.self), [2, 2, 2])
                XCTAssertEqual(gpu.asArray(Float.self), [3, 3, 3])
                return streams
            }
            if let previous {
                XCTAssertEqual(streams.0, previous.0)
                XCTAssertEqual(streams.1, previous.1)
                XCTAssertEqual(streams.2, previous.2)
            }
            previous = streams
            context.synchronize()
        }
    }

    func testNestedReusableContextsRestoreStreamsAfterThrowing() async {
        let outer = MLX.Stream.Context()
        let inner = MLX.Stream.Context()
        await Stream.withDefaultStream(outer) {
            let original = StreamOrDevice.default.stream
            do {
                try await Stream.withDefaultStream(inner) {
                    XCTAssertNotEqual(StreamOrDevice.default.stream, original)
                    await Task.yield()
                    throw CancellationError()
                }
                XCTFail("Expected cancellation")
            } catch is CancellationError {
                XCTAssertEqual(StreamOrDevice.default.stream, original)
            } catch {
                XCTFail("Unexpected error: \(error)")
            }
        }
    }

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
