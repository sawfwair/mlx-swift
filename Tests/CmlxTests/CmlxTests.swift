// Copyright © 2024 Apple Inc.

//
//  Copyright © 2023 Apple. All rights reserved.
//

import Dispatch
import Foundation
import XCTest

@testable import Cmlx

private final class CompileTraceCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0

    func increment() {
        lock.withLock {
            value += 1
        }
    }

    func read() -> Int {
        lock.withLock { value }
    }
}

private final class CompileRunStatuses: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Int32] = []

    func append(_ value: Int32) {
        lock.withLock {
            values.append(value)
        }
    }

    func read() -> [Int32] {
        lock.withLock { values }
    }
}

private func tracedIdentity(
    _ result: UnsafeMutablePointer<mlx_vector_array>?,
    _ input: mlx_vector_array,
    _ payload: UnsafeMutableRawPointer?
) -> Int32 {
    let counter = Unmanaged<CompileTraceCounter>.fromOpaque(payload!).takeUnretainedValue()
    counter.increment()
    return mlx_vector_array_set(result, input)
}

private func releaseTraceCounter(_ payload: UnsafeMutableRawPointer?) {
    Unmanaged<CompileTraceCounter>.fromOpaque(payload!).release()
}

class CmlxTests: XCTestCase {

    func testMinimal() throws {
        // smoke test making sure we can build, link & call C api
        //
        // note: there are convenience wrappers in MLX + the entire
        // wrapping of the API in swift

        var data: [Float] = [1, 2, 3, 4, 5, 6]
        var shape: [Int32] = [2, 3]

        let arr = mlx_array_new_data(&data, &shape, 2, MLX_FLOAT32)
        defer { mlx_array_free(arr) }

        var str = mlx_string_new()
        mlx_array_tostring(&str, arr)
        defer { mlx_string_free(str) }
        let description = String(cString: mlx_string_data(str))

        print(description)
    }

    func testCompileCacheCanBeErasedFromAnotherThread() throws {
        let counter = CompileTraceCounter()
        let statuses = CompileRunStatuses()
        let firstTraceFinished = DispatchSemaphore(value: 0)
        let cacheWasErased = DispatchSemaphore(value: 0)
        let workerFinished = expectation(description: "compile worker finished")
        let functionID: UInt = 0xC0_32_01

        Thread {
            var data: [Float] = [1]
            var shape: [Int32] = [1]
            var inputArray = mlx_array_new_data(&data, &shape, 1, MLX_FLOAT32)
            let inputs = mlx_vector_array_new_data(&inputArray, 1)
            let payload = Unmanaged.passRetained(counter).toOpaque()
            let function = mlx_closure_new_func_payload(
                tracedIdentity, payload, releaseTraceCounter)

            func compileAndApply() -> Int32 {
                var compiled = mlx_closure_new()
                var output = mlx_vector_array_new()
                defer {
                    mlx_vector_array_free(output)
                    mlx_closure_free(compiled)
                }

                var status = mlx_detail_compile(
                    &compiled, function, functionID, false, nil, 0)
                if status == 0 {
                    status = mlx_closure_apply(&output, compiled, inputs)
                }
                if status == 0 {
                    status = mlx_eval(output)
                }
                return status
            }

            statuses.append(compileAndApply())
            firstTraceFinished.signal()
            if cacheWasErased.wait(timeout: .now() + 10) != .success {
                statuses.append(-1)
            }
            statuses.append(compileAndApply())

            mlx_detail_compile_erase(functionID)
            mlx_closure_free(function)
            mlx_vector_array_free(inputs)
            mlx_array_free(inputArray)
            workerFinished.fulfill()
        }.start()

        XCTAssertEqual(firstTraceFinished.wait(timeout: .now() + 10), .success)
        XCTAssertEqual(mlx_detail_compile_erase(functionID), 0)
        cacheWasErased.signal()
        wait(for: [workerFinished], timeout: 10)

        XCTAssertEqual(statuses.read(), [0, 0])
        XCTAssertEqual(counter.read(), 2)
    }

}
