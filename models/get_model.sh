https://www.milk-island.net/translate/ggd/drive/api/v3/manage-downloads.html
これはbashのファイルなので下のスクリプトを用いるpythonファイルを作成する必要あり

file_id = '0BwwA4oUTeiV1UVNwOHItT0xfa2M'
request = drive_service.files().get_media(fileId=file_id)
fh = io.BytesIO()
downloader = MediaIoBaseDownload(fh, request)
done = False
while done is False:
    status, done = downloader.next_chunk()
    print "Download %d%%." % int(status.progress() * 100)