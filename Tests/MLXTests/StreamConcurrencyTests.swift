import MLX
import XCTest

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
}
