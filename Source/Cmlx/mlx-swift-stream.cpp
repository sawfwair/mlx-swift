#include "mlx/c/error.h"
#include "mlx/c/private/device.h"
#include "mlx/c/private/stream.h"
#include "mlx/stream.h"

extern "C" mlx_stream mlx_stream_new_thread_unsafe_device(mlx_device dev) {
  try {
    return mlx_stream_new_(
        mlx::core::new_thread_unsafe_stream(mlx_device_get_(dev)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_stream_new_();
  }
}
