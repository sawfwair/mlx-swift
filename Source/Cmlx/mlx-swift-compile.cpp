// Copyright © 2026 MLX Swift contributors.

#include "mlx/c/compile.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/compile_impl.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

using mlx::core::detail::CompileCacheWeakPtr;

class CompileCacheRegistry {
public:
  void remember(uintptr_t fun_id, const CompileCacheWeakPtr &cache) {
    std::lock_guard lock(mutex_);
    auto &caches = caches_[fun_id];
    caches.erase(std::remove_if(
                     caches.begin(), caches.end(),
                     [](const auto &candidate) { return candidate.expired(); }),
                 caches.end());

    auto same_cache = [&cache](const auto &candidate) {
      return !candidate.owner_before(cache) && !cache.owner_before(candidate);
    };
    if (std::none_of(caches.begin(), caches.end(), same_cache)) {
      caches.push_back(cache);
    }
  }

  std::vector<CompileCacheWeakPtr> take(uintptr_t fun_id) {
    std::lock_guard lock(mutex_);
    auto found = caches_.find(fun_id);
    if (found == caches_.end()) {
      return {};
    }

    auto caches = std::move(found->second);
    caches_.erase(found);
    return caches;
  }

private:
  std::mutex mutex_;
  std::unordered_map<uintptr_t, std::vector<CompileCacheWeakPtr>> caches_;
};

CompileCacheRegistry &compile_cache_registry() {
  // The registry can be reached from Swift finalizers during process teardown.
  static auto *registry = new CompileCacheRegistry();
  return *registry;
}

} // namespace

extern "C" int mlx_compile(mlx_closure *res, const mlx_closure fun,
                           bool shapeless) {
  try {
    mlx_closure_set_(*res,
                     mlx::core::compile(mlx_closure_get_(fun), shapeless));
  } catch (std::exception &e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_detail_compile(mlx_closure *res, const mlx_closure fun,
                                  uintptr_t fun_id, bool shapeless,
                                  const uint64_t *constants,
                                  size_t constants_num) {
  try {
    auto cache = mlx::core::detail::compile_cache();
    compile_cache_registry().remember(fun_id, cache);
    mlx_closure_set_(
        *res, mlx::core::detail::compile(
                  mlx_closure_get_(fun), fun_id, shapeless,
                  std::vector<uint64_t>(constants, constants + constants_num)));
  } catch (std::exception &e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_detail_compile_clear_cache(void) {
  try {
    mlx::core::detail::compile_clear_cache(mlx::core::detail::compile_cache());
  } catch (std::exception &e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_detail_compile_erase(uintptr_t fun_id) {
  try {
    for (const auto &cache : compile_cache_registry().take(fun_id)) {
      mlx::core::detail::compile_erase(cache, fun_id);
    }
  } catch (std::exception &e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_disable_compile(void) {
  try {
    mlx::core::disable_compile();
  } catch (std::exception &e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_enable_compile(void) {
  try {
    mlx::core::enable_compile();
  } catch (std::exception &e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_set_compile_mode(mlx_compile_mode mode) {
  try {
    mlx::core::set_compile_mode(mlx_compile_mode_to_cpp(mode));
  } catch (std::exception &e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
