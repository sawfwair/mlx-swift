// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Parameter type for all MLX operations.
///
/// Use this to control where operations are evaluated:
///
/// ```swift
/// // produced on cpu
/// let a = MLXRandom.uniform([100, 100], stream: .cpu)
///
/// // produced on gpu
/// let b = MLXRandom.uniform([100, 100], stream: .gpu)
/// ```
///
/// If omitted it will use the ``default``, which will be ``Device/gpu`` unless
/// set otherwise.
///
/// ### See Also
/// - <doc:using-streams>
/// - ``Stream``
/// - ``Device``
public struct StreamOrDevice: Sendable, CustomStringConvertible, Equatable {

    public let stream: Stream

    private init(_ stream: Stream) {
        self.stream = stream
    }

    /// The default stream on the default device.
    ///
    /// This will be ``Device/gpu`` unless ``Device/setDefault(device:)``
    /// sets it otherwise.
    public static var `default`: StreamOrDevice {
        StreamOrDevice(Stream.defaultStream ?? Device.defaultStream())
    }

    public static func device(_ device: Device) -> StreamOrDevice {
        StreamOrDevice(Stream.defaultStream(device))
    }

    /// The ``Stream/defaultStream(_:)`` on the ``Device/cpu``
    public static var cpu: StreamOrDevice {
        device(.cpu)
    }

    /// The ``Stream/defaultStream(_:)`` on the ``Device/gpu``
    ///
    /// ### See Also
    /// - ``GPU``
    public static var gpu: StreamOrDevice {
        device(.gpu)
    }

    public static func stream(_ stream: Stream) -> StreamOrDevice {
        StreamOrDevice(stream)
    }

    /// Internal context -- used with Cmlx calls.
    public var ctx: mlx_stream {
        stream.ctx
    }

    public var description: String {
        stream.description
    }
}

/// A stream of evaluation attached to a particular device.
///
/// Typically this is used via the `stream: ` parameter on a method with a ``StreamOrDevice``:
///
/// ```swift
/// let a: MLXArray ...
/// let result = sqrt(a, stream: .gpu)
/// ```
///
/// Read more at <doc:using-streams>.
///
/// ### See Also
/// - <doc:using-streams>
/// - ``StreamOrDevice``
public final class Stream: @unchecked Sendable, Equatable {

    let ctx: mlx_stream

    public static var gpu: Stream {
        Stream(mlx_default_gpu_stream_new())
    }

    public static var cpu: Stream {
        Stream(mlx_default_cpu_stream_new())
    }

    @TaskLocal static var defaultStream: Stream?
    @TaskLocal static var defaultCPUStream: Stream?
    @TaskLocal static var defaultGPUStream: Stream?

    /// Reusable CPU and GPU streams for sequential asynchronous graph work.
    ///
    /// Keep one context per active operation, and reuse it for later operations
    /// with ``Stream/withDefaultStream(_:isolation:_:)``. MLX retains backend streams
    /// for the lifetime of the process, so creating a new context for every
    /// request can accumulate command queues in a long-running server.
    ///
    /// The caller must serialize graph construction and evaluation on a context,
    /// including evaluation of lazy arrays returned from an earlier operation.
    public final class Context: Sendable {
        fileprivate let cpu: Stream
        fileprivate let gpu: Stream?
        fileprivate let selected: Stream

        /// Creates streams and selects the default device for unqualified work.
        ///
        /// - Parameter device: The device to select, or the current default device.
        public init(device: Device? = nil) {
            let device = device ?? Device.defaultDevice()
            let cpu = Stream(threadUnsafe: .cpu)
            let gpu = Stream.threadUnsafeStreamIfAvailable(.gpu)
            self.cpu = cpu
            self.gpu = gpu
            switch device.deviceType {
            case .cpu:
                self.selected = cpu
            case .gpu:
                self.selected = gpu ?? Stream(threadUnsafe: .gpu)
            default:
                fatalError("Unexpected device type: \(device)")
            }
        }

        /// Wait for submitted CPU and GPU work before reusing this context.
        public func synchronize() {
            cpu.synchronize()
            gpu?.synchronize()
        }
    }

    /// Use an existing context's task-local streams without allocating new ones.
    public static func withDefaultStream<R>(
        _ context: Context,
        isolation _: isolated (any Actor)? = #isolation,
        _ body: () async throws -> R
    ) async rethrows -> R {
        try await $defaultCPUStream.withValue(context.cpu) {
            try await $defaultGPUStream.withValue(context.gpu) {
                try await $defaultStream.withValue(context.selected, operation: body)
            }
        }
    }

    /// Set the ``StreamOrDevice/default`` scoped to a Task.
    public static func withNewDefaultStream<R>(device: Device? = nil, _ body: () throws -> R)
        rethrows -> R
    {
        let device = device ?? Device.defaultDevice()
        return try $defaultStream.withValue(Stream(device), operation: body)
    }

    /// Set the ``StreamOrDevice/default`` scoped to a Task.
    public static func withNewDefaultStream<R>(
        device: Device? = nil,
        isolation _: isolated (any Actor)? = #isolation,
        _ body: () async throws -> R
    ) async rethrows -> R {
        try await withDefaultStream(Context(device: device), body)
    }

    init(_ ctx: mlx_stream) {
        self.ctx = ctx
    }

    /// Default stream on the default device.
    public init() {
        let device = Device.defaultDevice()
        var ctx = mlx_stream_new()
        mlx_get_default_stream(&ctx, device.ctx)
        self.ctx = ctx
    }

    @available(*, deprecated, message: "use init(Device) -- index not supported")
    public init(index: Int32, _ device: Device) {
        self.ctx = evalLock.withLock {
            mlx_stream_new_device(device.ctx)
        }
    }

    /// New stream on the given device.
    ///
    /// See also ``withNewDefaultStream(device:_:)``
    public init(_ device: Device) {
        self.ctx = evalLock.withLock {
            mlx_stream_new_device(device.ctx)
        }
    }

    /// A stream for a sequential graph whose Swift task may resume on a
    /// different executor thread after suspension.
    private init(threadUnsafe device: Device) {
        self.ctx = evalLock.withLock {
            mlx_stream_new_thread_unsafe_device(device.ctx)
        }
    }

    private static func threadUnsafeStreamIfAvailable(_ device: Device) -> Stream? {
        var available = false
        mlx_device_is_available(&available, device.ctx)
        return available ? Stream(threadUnsafe: device) : nil
    }

    deinit {
        _ = evalLock.withLock {
            mlx_stream_free(ctx)
        }
    }

    /// Synchronize with the given stream
    public func synchronize() {
        _ = evalLock.withLock {
            mlx_synchronize(ctx)
        }
    }

    static public func defaultStream(_ device: Device) -> Stream {
        switch device.deviceType {
        case .cpu: defaultCPUStream ?? .cpu
        case .gpu: defaultGPUStream ?? .gpu
        default: fatalError("Unexpected device type: \(device)")
        }
    }

    public static func == (lhs: Stream, rhs: Stream) -> Bool {
        mlx_stream_equal(lhs.ctx, rhs.ctx)
    }
}

extension Stream: CustomStringConvertible {
    public var description: String {
        var s = mlx_string_new()
        defer { mlx_string_free(s) }
        _ = evalLock.withLock {
            mlx_stream_tostring(&s, ctx)
        }
        return String(cString: mlx_string_data(s), encoding: .utf8)!
    }
}
