@echo off
echo Starting Cloud Detection System...
echo.

REM Set environment variables to fix Windows issues
set KMP_DUPLICATE_LIB_OK=TRUE
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll

REM Launch Streamlit with polling file watcher (Windows-safe)
streamlit run streamlit_app.py --server.fileWatcherType=poll

pause
