# Proyecto MICROESTRUCTURA Y SISTEMAS DE TRADING

## Configuración y Ejecución del Proyecto

1.  Creacion y activacion del entorno virtual:

    **Para sistemas basados en Unix/Linux o macOS:**
      ```bash
      python -m venv venv
      ```
      ```bash
      source venv/bin/activate
      ```

    **Para sistemas basados en Windows:**
       ```bash
       python -m venv venv
       ```
       ```bash
       .\venv\Scripts\activate
       ```
2. Instalar dependencias
    ```bash
    pip install -r requirements.txt
    ```
## Ejecución del Proyecto

1. Abre el Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

2. Abre el archivo `report.ipynb` y ejecuta las celdas para ver los resultados.



## Configuración Inicial para la modificacion del proyecto

Antes de comenzar, configura tu nombre de usuario y dirección de correo electrónico para Git:

**NOTA: Si tienes PyCharm puedes vincular tu cuenta de GitHub para ahorrarte este paso y puedes clonar desde PyCharm**

```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"
```
## Para modificar el proyecto, sigue estos pasos:

1. **Crear un Nuevo Branch:**
   Antes de empezar a trabajar en una nueva función o corrección de errores, crea un nuevo branch:

    ```bash
    git checkout -b nombre-del-branch
    ```

2. **Ver Branches Existente:**
   Para ver la lista de branches existentes en tu repositorio local:

    ```bash
    git branch
    ```

   Si deseas ver información adicional sobre los branches, como los últimos commits en cada branch:

    ```bash
    git show-branch
    ```

3. **Cambiar a un Branch Existente:**
   Si ya hay un branch creado y deseas trabajar en él:

    ```bash
    git checkout nombre-del-branch
    ```

## Realiza tus cambios y haz commit

1. **Haz los cambios necesarios en tu código.**
    Añade los cambios al área de preparación y realiza un commit con un mensaje descriptivo:

    ```bash
    git add .
    git commit -m "Descripción de los cambios"
    ```

## Sincronizar con el Repositorio Remoto

1. **Si otros colaboradores han realizado cambios en el repositorio remoto, es recomendable realizar un pull antes de realizar un push:**

    ```bash
    git pull origin nombre-del-branch
    ```

2. **Para subir tus cambios al repositorio remoto:**

    ```bash
    git push origin nombre-del-branch
    ```

## Verificar el Estado de tus Cambios

1. Puedes usar `git status` para ver el estado actual de tus cambios en el repositorio local. Esto te proporcionará información sobre archivos modificados, archivos en el área de preparación (staging) y otros detalles relacionados con tu trabajo actual.

    ```bash
    git status
    ```



