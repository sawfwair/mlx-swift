#include "mlx/c/mlx.h"
#include "mlx/c/transforms_impl.h"
#include "mlx/c/linalg.h"
#include "mlx/c/fast.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns a stream that can build a sequential graph across Swift concurrency
 * executor threads.
 */
mlx_stream mlx_stream_new_thread_unsafe_device(mlx_device dev);

/** Compile with cache identities retained across Swift executor hops. */
int mlx_swift_detail_compile(mlx_closure* res, const mlx_closure fun,
    uintptr_t fun_id, bool shapeless, const uint64_t* constants, size_t constants_num);
/** Erase a Swift function from all of its originating thread-local caches. */
int mlx_swift_detail_compile_erase(uintptr_t fun_id);

#ifdef __cplusplus
}
#endif
