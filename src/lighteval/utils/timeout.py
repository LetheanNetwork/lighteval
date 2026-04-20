# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import threading


def timeout(timeout_seconds: int = 10):  # noqa: C901
    """A decorator that applies a timeout to the decorated function.

    Args:
        timeout_seconds (int): Number of seconds before timing out the decorated function.
            Defaults to 10 seconds.

    Notes:
        On Unix systems, uses a signal-based alarm approach which is more efficient as it doesn't require spawning a new process.
        However, signal.signal only works in the main thread. If called from a sub-thread, it falls back to a timer approach.
        On Windows systems, uses a multiprocessing-based approach since signal.alarm is not available. This will incur a huge performance penalty.

    Returns:
        Callable: A decorator function that wraps the original function with timeout functionality
    """
    if os.name == "posix":
        # Unix-like approach: signal.alarm or Timer thread if not in main thread
        import signal

        def decorator(func):
            def handler(signum, frame):
                raise TimeoutError("Operation timed out!")

            def wrapper(*args, **kwargs):
                if threading.current_thread() is threading.main_thread():
                    # We are in the main thread, we can use signals
                    old_handler = signal.getsignal(signal.SIGALRM)
                    signal.signal(signal.SIGALRM, handler)
                    signal.alarm(timeout_seconds)
                    try:
                        return func(*args, **kwargs)
                    finally:
                        # Cancel the alarm and restore previous handler
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                else:
                    # We are in a subthread, signals don't work. Use a timer that raises an exception in this thread.
                    # Since ctypes is messy, a simple but suboptimal fallback is to use the Thread/Queue approach like Windows.
                    from queue import Queue
                    from threading import Thread

                    q = Queue()

                    def run_func(q, args, kwargs):
                        try:
                            result = func(*args, **kwargs)
                            q.put((True, result))
                        except Exception as e:
                            q.put((False, e))

                    t = Thread(target=run_func, args=(q, args, kwargs))
                    t.start()
                    t.join(timeout_seconds)

                    if t.is_alive():
                        # We cannot terminate a thread easily in python, so we just raise TimeoutError and leave the thread hanging
                        raise TimeoutError("Operation timed out (in sub-thread)!")

                    success, value = q.get()
                    if success:
                        return value
                    else:
                        raise value

            return wrapper

        return decorator

    else:
        # Windows approach: use multiprocessing
        from multiprocessing import Process, Queue

        def decorator(func):
            def wrapper(*args, **kwargs):
                q = Queue()

                def run_func(q, args, kwargs):
                    try:
                        result = func(*args, **kwargs)
                        q.put((True, result))
                    except Exception as e:
                        q.put((False, e))

                p = Process(target=run_func, args=(q, args, kwargs))
                p.start()
                p.join(timeout_seconds)

                if p.is_alive():
                    # Timeout: Terminate the process
                    p.terminate()
                    p.join()
                    raise TimeoutError("Operation timed out!")

                # If we got here, the process completed in time.
                success, value = q.get()
                if success:
                    return value
                else:
                    # The child raised an exception; re-raise it here
                    raise value

            return wrapper

        return decorator
