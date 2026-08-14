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

#ifdef __cplusplus
}
#endif
