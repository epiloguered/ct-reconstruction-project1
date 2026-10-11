# Windows原生Word：更新目录、统一字体并保存，随后导出QA PDF。
$ErrorActionPreference = 'Stop'
$reportRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$reportFile = Join-Path $reportRoot 'report\CT图像重建与稀疏角度优化实验报告修订版.docx'
$renderFolder = Join-Path $reportRoot 'outputs\report_build\render'
New-Item -ItemType Directory -Path $renderFolder -Force | Out-Null
$renderFile = Join-Path $renderFolder 'report.pdf'
$reportWord = $null
$reportDoc = $null
try {
    $reportWord = New-Object -ComObject Word.Application
    $reportWord.Visible = $false
    $reportWord.DisplayAlerts = 0
    $reportDoc = $reportWord.Documents.Open($reportFile, $false, $false, $false)
    if ($reportDoc.ReadOnly) { throw '报告被其他程序锁定，无法保存更新后的目录。请关闭该报告后重试。' }
    $tocRange = $reportDoc.Content
    if ($tocRange.Find.Execute('[[TOC]]')) {
        $tocRange.Text = ''
        $null = $reportDoc.TablesOfContents.Add($tocRange, $true, 1, 1)
    }
    foreach ($reportToc in $reportDoc.TablesOfContents) { $reportToc.Update() }
    $reportDoc.Content.Font.NameFarEast = '宋体'
    $reportDoc.Content.Font.NameAscii = 'Times New Roman'
    $reportDoc.Content.Font.NameOther = 'Times New Roman'
    foreach ($reportToc in $reportDoc.TablesOfContents) {
        $reportToc.Range.Font.Size = 10.5
        $reportToc.Range.ParagraphFormat.SpaceAfter = 6
        $reportToc.Range.ParagraphFormat.LineSpacingRule = 0
    }
    $reportDoc.Repaginate()
    foreach ($reportToc in $reportDoc.TablesOfContents) { $reportToc.UpdatePageNumbers() }
    $reportDoc.Save()
    $pageCount = $reportDoc.ComputeStatistics(2)
    $reportDoc.ExportAsFixedFormat($renderFile, 17)
    Write-Output "Native Word export succeeded; pages=$pageCount; PDF=$renderFile"
} finally {
    if ($null -ne $reportDoc) { $reportDoc.Close(0) }
    if ($null -ne $reportWord -and $reportWord.Documents.Count -eq 0) { $reportWord.Quit() }
}
