SELECT
    p.ProductKey,
    p.StandardCost,
    p.ListPrice AS Price
FROM
    DimProduct AS p
WHERE
    p.ProductKey IS NOT NULL 
	AND p.StandardCost IS NOT NULL
	AND p.ListPrice IS NOT NULL;