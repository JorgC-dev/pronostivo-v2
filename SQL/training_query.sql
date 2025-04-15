SELECT
[F].[Fecha] AS fecha,
SUM([H]. [cantidad]) AS TotalVentas
FROM
[demo_prediccion]. [dbo]. [hechos] AS [H]
INNER JOIN [demo_prediccion]. [dbo]. [Dim_fechas] AS [f] ON [H].[id_DimFechas] = [F].[id]
WHERE 
[F].[Fecha] > '2016-12-31' AND [F].[Fecha] < '2020-01-01'
GROUP BY [F].[Fecha]
ORDER BY [F].[Fecha]